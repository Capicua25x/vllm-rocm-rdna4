# vLLM 0.26.1 — RDNA4 port (gfx1200/gfx1201: Radeon AI PRO R9700, RX 9070 XT)

Branch `rdna4-port-0.26.1` = upstream vLLM (0.26.1 line, base commit `99e62b802`) + the changes needed
to serve well on AMD RDNA4 consumer/workstation GPUs, which are outside the official ROCm vLLM targets
(gfx90a/942/950).

> **⚡ 2026-08-20 — major performance upgrade (rc10):** tuned per-shape GEMM configs for the R9700
> (regenerated with vLLM's own tuner) ship in-tree. Short-prompt c32 aggregate: FP8 arm **754 → 1,014
> tok/s (+34%)**, MXFP4 arm **579 → 713 (+23%)**; every cell improved. Image: `capicua25x/vllm-rocm-rdna4:latest`.
> Credit where due: this port was always geared to **concurrent serving** (32 sequences, speculative
> decoding, accuracy-gated) — the single-stream tuning insight came from the community:
> [andysalerno](https://github.com/andysalerno/R9700-serving)'s and
> [prcoe1](https://github.com/prcoe1/r9700-serving)'s benchmarks surfaced the untuned-GEMM gap; folding
> that lever into the concurrency stack closed the loop.
>
> **2026-08-24 measurement update (bench v4):** the essay bench behind these absolutes reused one
> fixed prompt, which lets the stateful spec-decode drafter partially replay earlier generations —
> inflating absolute essay tok/s (shape-dependent on re-measurement: ≈10-20% on most greedy-raw cells, up to ≈2x on the worst (a GB10-pair short cell), while one sampled-path 6k cell even measured slightly higher under v4 (different sampling provenance + build drift)). The **relative** rc10 gains above were measured
> like-for-like and **stand**; the absolute cells are replay-era. Honest v4 short-prompt c32
> aggregate on the FP8 arm: **789 tok/s** (rotating topics + per-invocation nonce + temp 0.7 —
> nothing is ever regenerated). Fixed in
> [Capicua25x/modelbench](https://github.com/Capicua25x/modelbench) v4; honest tables in the
> "Measurement update 2026-08-24" block of the Throughput section below.

**Carriable fixes, each its own branch:** `fix/suppress-stops-in-reasoning` (detokenizer guard +
CPU test — vllm-project/vllm#53066) · `fix/unified-attn-3d-smallq` (3D split-KV gate for spec-decode
verify shapes) · `feat/rdna4-mxfp4-linear` (MXFP4 dense/MoE path). Anyone may carry them upstream;
authorship preservation appreciated.

## Attribution and lineage

**This port stands on Rob Smith's (`tcclaviger`) RDNA4 work, and without it none of this exists.**
He did the gfx1201 enablement first, for the vLLM **0.18.1** line — the working RDNA4 base of
gfx1201 hipBLASLt plus MXFP4/NVFP4 MoE kernels — and that is the foundation everything here is
built on. Chain: his 0.18.1 work → forward-port to **0.19.1** in
`Capicua25x/vllm-rocm-rdna4-legacy` (archived) → this **0.26.1** branch. Separately,
`rdna_fp8.py` follows the design of his `_matmul_fp8_ogs` (0.24 line, W8A8): the same per-K-group
scale fold on WMMA v2, applied to the fp32 accumulator after the dot. That credit is carried in the
file itself, not only here.

His RDNA4 work is distributed as Docker images under `tcclaviger`.

What is **this project's** own work, scoped to what is actually in this branch:

* the MXFP4 nibble→e4m3 unpack and the three-regime dispatch in `rdna_fp8.py`;
* a **gate relaxation** on upstream's 3D split-KV path so small-q spec-decode verify shapes can use it
  (`MAX_QLEN_3D`) — 11 lines on a kernel by Burkhard Ringlein, Jan van Lunteren, Chih-Chieh Yang and
  Thomas Parnell. Arrived at independently, but equivalent relaxations were proposed upstream **first**
  (vllm-project/vllm #44652, #45450, #46724 — none merged as of 2026-08);
* the RDNA4 gating in `mxfp4_utils.py` and `compressed_tensors_moe_w4a4_mxfp4.py`;
* model bring-up, the quantization recipes below, and the validation campaigns.

**Forward-ported rather than original**, and derived from the lineage above: the weight-only
`RdnaMxfp4LinearKernel`, the RDNA4 branch in `_swizzle_mxfp4`, the MXFP4 MoE routing, and — the largest
borrowed piece — the RDNA4 `triton_kernels` graft the whole MXFP4 path depends on (`RDNAMXValueLayout` /
`mxfp4_dequant_rdna`, plus the RDNA branches in `_matmul_ogs.py`, `opt_flags.py`, `target_info.py`). That
graft is **not** in upstream `triton_kernels`; it is not carried in this source repository, but the
published image ships it. See `NOTICE`.

Carried **outside this branch**, and so not claimed as part of it: the ROCm fp8-KV attention overlay
(applied at container build time) and the clean-room head-dim-512 flash-prefill kernel (blueprint
from llama.cpp, MIT, no code copied), which belongs to the 0.19.1 line.

Four pre-existing upstream files are modified here; each carries a modification notice under its
SPDX header, per Apache-2.0 §4(b). Everything else in the diff is newly added. See `NOTICE`.

## Quantization policy (2026-08-16)

**4-bit: AMD Quark / MXFP4 only.** Everything quantized in-house goes through Quark and ships as MXFP4; the RTN
pipeline is retired for new work. **NVFP4 is not carried on this port** — the 0.19.1 implementation was not
forward-ported and is not maintained. RDNA4 has no FP4 datapath either way, so the format difference is scale
handling inside one kernel, not a separate port.

**8-bit: the vendor's own FP8** (e.g. `Qwen/Qwen3.8-27B-FP8`) — native FP8 WMMA on gfx1201, no in-house work needed.

**Consumption is unrestricted**: the port loads what the ecosystem publishes (Quark, compressed-tensors, AWQ/GPTQ,
vendor FP8). The policy above is about what this project *produces*.

## Quantization recipe (Quark MXFP4)

This section is the recipe of record for the MXFP4 checkpoints this project builds. The older
`Capicua25x/qwen3.6-mxfp4-rdna4` repo documents the **retired RTN (`olka/qstream`) pipeline** and a
different model; it is kept for history and is *not* the recipe for anything built after 2026-08-16.

**Toolkit:** AMD Quark 0.12.post1 (`amd-quark==0.12.post1`, python 3.13, torch 2.10.0+rocm7.0).
Scheme `mxfp4`: weights fp4, `per_group`, `group_size 32`, E8M0 scales, round half-even, static;
export `weight_format=real_quantized`, `pack_method=reorder`, `quant_mode=eager_mode`.
All builds are **data-free** (`algo_config=null`, `QUARK_ALGO=none` — RTN-style, *not* AWQ) and run on
**CPU**, so the GPUs stay free for benchmarking. Packing is genuine 4-bit: a bf16 `[out, in]` weight
becomes a U8 `[out, in/2]` plus a U8 `weight_scale` `[out, in/32]`.

**Prerequisite: `ninja` on `PATH`.** Importing `quark.torch` JIT-builds a C++ hw-emulation extension, so
without ninja the *import itself* dies — `RuntimeError: Ninja is required to load C++ extensions` — before
any driver code runs. `pip install ninja` drops the binary in the venv's `bin/`, which is **not** on `PATH`
when the interpreter is invoked by absolute path. Export it:
`PATH=<venv>/bin:$PATH <venv>/bin/python driver.py`.

### The policy

Only **MLP / MoE-expert projections** go to 4 bit. Attention (q/k/v/o and any per-head gate), every
norm, embeddings, `lm_head`, MoE routers and shared-expert gates, `conv1d` in linear-attention layers,
the whole vision path, and any MTP/draft head stay **bf16**.

Two consequences worth stating plainly, because both are easy to misread:

* **The checkpoints declare W4A4, not weight-only.** Quark's `mxfp4` scheme turns on dynamic fp4
  activation quantization by default and we did not override it, so every build carries
  `global_quant_config.input_tensors = {dtype: fp4, is_dynamic: true, per_group, group_size 32, e8m0}`.
  On this port that declaration is **not honoured**: the weight-only kernel ignores activation quant,
  and the rc6 kernel uses its own per-(token, 32-K-group) dynamic e4m3. Read the difference from
  AMD-style whole-decoder builds as **coverage**, not activation width.
* **Coverage is the only difference from `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`,** which quantizes the
  decoder's attention too: 496 quantized modules there vs **432** here on the same model, the delta
  being exactly the 16 full-attention layers' q/k/v/o. (AMD's build also ships its MTP head in bf16
  while omitting `mtp.*` from `exclude`, so it needs a 15-entry config patch before vLLM will load it
  with spec-decode.)

### Two build paths

* **`ModelQuantizer.direct_quantize_checkpoint(...)`** — file-to-file, reads the snapshot directory and
  writes the quantized one, never materializing the model. Used for five of six builds. Ships a
  `model.safetensors.index.json`. Fast: ≈4 min for the 27B on CPU; the 35B MoE stage took ≈18.6 min
  including a 70 GB download.
* **`preprocess_for_quantization` → `quantize_model` → `freeze` → `export_safetensors`** — required when
  the checkpoint stores experts as **stacked 3D tensors** (Gemma-4 MoE). The preprocess step explodes
  them into per-expert Linears and frees the fused source, which is both what Quark needs to see them
  and what vLLM's loader wants on the other end (`gemma4.py` accepts "already per-expert 2D weights (if
  quantized)"). File-to-file quantized **0 of 3,840** expert modules before this path was used.
  Caveat: this exporter writes a **single `model.safetensors` with no index file** (≈206 s to quantize,
  ≈5 s to export for the 26B-A4B).

Note that `direct_quantize_checkpoint` reads `config.json` off the filesystem — it does not resolve HF
repo ids. Snapshot first, hand it the local directory.

Minimal complete driver for the file-to-file path (this is the whole config construction — the
`LLMTemplate` step is not optional: Quark's built-in template list does not cover `qwen3_5`, and
`LLMTemplate.get()` raises for an unregistered model type):

```python
import transformers
from quark.torch import LLMTemplate, ModelQuantizer

SRC = "<local snapshot dir>"   # a directory — direct_quantize_checkpoint does not resolve repo ids
OUT = "<output dir>"

EXCLUDE = [                    # Qwen3.8-27B, 15 globs — see "Per-family exclude lists" below
    "lm_head", "*embed_tokens*",
    "*.self_attn.q_proj", "*.self_attn.k_proj", "*.self_attn.v_proj", "*.self_attn.o_proj",
    "*.self_attn.q_norm", "*.self_attn.k_norm",
    "*norm*",
    "*.linear_attn.conv1d", "*.linear_attn.norm",
    "*.mlp.gate",
    "mtp*",
    "*visual*", "*vision*",
]

model_type = transformers.AutoConfig.from_pretrained(SRC, trust_remote_code=True).model_type

if model_type not in LLMTemplate.list_available():
    LLMTemplate.register_template(LLMTemplate(
        model_type=model_type,
        kv_layers_name=["*language_model.*k_proj", "*language_model.*v_proj"],
        q_layer_name="*language_model.*q_proj",
        exclude_layers_name=EXCLUDE,
    ))

# No `algorithm=` kwarg -> `algo_config` stays null -> data-free, CPU-only.
# Pass `algorithm="awq"` instead for the activation-aware variant (needs a GPU and forward passes).
quant_config = LLMTemplate.get(model_type).get_config(scheme="mxfp4", exclude_layers=EXCLUDE)

ModelQuantizer(quant_config).direct_quantize_checkpoint(
    pretrained_model_path=SRC, save_path=OUT, device="cpu")
```

`get_config` logs `Expanding exclude pattern [X] to [X, X.*]` for every glob that could name a parent
module, so the 15 globs become 30 entries on the config object. That is Quark being thorough, not a
misparse. For the stacked-3D-expert models substitute the second build path
(`preprocess_for_quantization` → `quantize_model` → `freeze` → `export_safetensors`) for the
`direct_quantize_checkpoint` call; everything above it is unchanged.

### Per-family exclude lists

**None of these transfer between families.** They are derived per model from that repo's
`model.safetensors.index.json`, because the module naming differs materially. The globs below are the
driver input; the number in brackets is how many concrete module names Quark recorded in the artifact's
`quantization_config.exclude` (only quantizable Linear/Conv modules are recorded, so `*norm*` and
`*embed_tokens*` frequently contribute **zero** recorded entries — do not read their absence as the
pattern not having applied). Patterns are matched with `fnmatch.fnmatch` — in
`file2file_quantization.py` for the file-to-file path and in `model_transformation.py` for the in-memory
one — so **brace expansion is not available**: `*.self_attn.{q,k,v,o}_proj` matches nothing at all
(fnmatch escapes the braces into a literal), and the modules it was meant to protect get quantized
silently. Enumerate alternatives literally.

*Qwen3.8-27B* — `qwen3_5`, dense hybrid VL — 15 globs → **231** recorded:
```
lm_head, *embed_tokens*
*.self_attn.q_proj, *.self_attn.k_proj, *.self_attn.v_proj, *.self_attn.o_proj
*.self_attn.q_norm, *.self_attn.k_norm
*norm*
*.linear_attn.conv1d, *.linear_attn.norm
*.mlp.gate                 # MoE router name — inert on this dense model, kept for symmetry
mtp*                       # whole MTP head: vLLM needs it unquantized for spec-decode
*visual*, *vision*         # vision tower + merger + patch embed
```
*Ornith-1.0-35B* — `qwen3_5_moe` — the same list **plus** `*.mlp.shared_expert_gate` (16 globs).
After the MTP graft (below) the recorded list is **1046** = 261 + the 785 grafted `mtp.*` modules named
explicitly.

*Muse-Glimmer-30B* — `muse_glimmer`, dense VL — 7 globs → **564** recorded:
```
lm_head, *embed_tokens*
*self_attn*                # glob, NOT enumerated q/k/v/o — see quirk
*norm*
*vision_tower*, *vision_adapter*, *vision_projection*
```

*Mistral-Small-3.2-24B* — `mistral3`, dense — 6 globs → **333** recorded:
```
*lm_head*, *embed_tokens*
*self_attn*, *norm*
*vision_tower*, *multi_modal_projector*
```

*gemma-4-31B-it* — `gemma4`, dense — 7 globs → **419** recorded:
```
lm_head, *embed_tokens*, *embed_vision*
*self_attn*, *layer_scalar*, *norm*, *vision_tower*
```

*gemma-4-26B-A4B-it* — `gemma4`, 128-expert MoE — 8 globs → **337** recorded: the 31B list **plus**
`*router*`.

### Family quirks (the part that does not transfer)

* **Gemma-4's MoE router is `router.proj` / `router.scale` / `router.per_expert_scale`, not Qwen's
  `mlp.gate`.** No tensor name in the checkpoint contains `.mlp.gate.` (only `.mlp.gate_proj.`, the
  SwiGLU gate), so a copied Qwen list would have matched nothing and silently quantized all 30 routers.
* **Muse-Glimmer has an extra `self_attn.gate_proj`** — a per-head output gate, `[4096, 6656]` — that an
  enumerated q/k/v/o exclude list leaves exposed. The `*self_attn*` glob is what catches it.
* **Mistral-Small nests as `language_model.model.layers.*`, not `model.language_model.*`,** and its
  vision tower names modules `attention.` / `feed_forward.` rather than `self_attn.` / `mlp.` — so
  `*self_attn*` does not reach the vision tower and `*vision_tower*` must carry it (169 of the 333
  recorded entries).
* **Gemma-4 carries a per-layer `layer_scalar`** that must stay bf16. It survives untouched — but it is
  a bare parameter, not an `nn.Linear`, so it never appears in the recorded exclude list and the
  `*layer_scalar*` pattern is probably inert. Kept as cheap insurance; **effect unverified** (no control
  run without it).
* **Gemma-4 has `v_proj` on only some layers** (`attention_k_eq_v`): 50 of 60 on the 31B, 25 of 30 on the
  26B-A4B — the missing ones are exactly the `full_attention` layers. That is why the 26B's recorded
  self-attn exclude count is 115, not 120; nothing went missing.
* **Ornith-1.0-35B declares `mtp_num_hidden_layers: 1` but ships no `mtp.*` tensors** (verified against
  the Hub: 0 of 31,666), so a plain requant serves with no speculative decoding. Its head is a
  **cross-model graft**: 785 bf16 `mtp.*` tensors from a sibling Qwen3.6-35B-A3B MoE checkpoint with the
  same hidden size and expert layout, added as one shard and named explicitly in `exclude`. The donor is
  [`pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`](https://huggingface.co/pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4)
  (**Apache-2.0**), which quantizes the trunk but keeps `mtp.*` bf16; the head itself is
  [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)'s own trained MTP block
  (**Apache-2.0**), relaid from 2 stacked 3D expert tensors into 256x3 per-expert 2D weights — 19 - 2 +
  768 = 785, and the 785 tensor names match the donor index exactly. These are third-party weights
  carried verbatim, so the donor repo and its Apache-2.0 notice must be stated on any card that ships
  it — it is not an Ornith-trained head. MTP verification is
  lossless by construction, so a mismatched head can only cost draft speed, never change outputs;
  **acceptance rate on this checkpoint is unmeasured.**
* **Qwen3.8-27B is the opposite case and needs no graft:** it declares *and ships* its own MTP head
  (15 bf16 tensors, 8 of them Linears caught by `mtp*`). Do not generalize Ornith's graft to the family.
* **AutoProcessor can fail to save** and the driver only prints the exception. It failed on every
  multimodal build for want of `torchvision` / `pillow` in the quant venv. Harmless for the gemma-4
  builds (upstream ships no `preprocessor_config.json` and the existing `processor_config.json` was
  restored); **not harmless for Mistral-Small**, whose artifact is consequently missing
  `preprocessor_config.json`, `processor_config.json`, `chat_template.jinja` and
  `special_tokens_map.json`, with a 176-byte stub `tokenizer_config.json`. The vision weights are intact;
  the config needed to feed them images is not there. Install both libraries before rebuilding.

### Verification method

Key on the **real Quark artifacts** — a module counts as quantized iff it has one of
`weight_scale` / `weight_packed` / `qweight` / `weight_zero_point`. Then check **both directions**:

1. *leakage* — nothing outside the intended set carries a quant artifact (attention, norms, embeddings,
   `lm_head`, routers/gates, `conv1d`, vision, MTP);
2. *over-exclusion* — every MLP/expert projection that should be 4-bit is 4-bit, per layer, with the
   tensor accounting closing exactly (e.g. the 27B: base 1199 tensors + 432 `weight_scale` = 1631 in the
   artifact, nothing dropped or renamed).

Three traps, all of which bit this project:

* **A bare `*_scale` suffix check gives false positives.** Gemma-4 ships `vision_tower.std_scale` and
  Gemma-4 MoE ships `router.scale` / `router.per_expert_scale` in the *original bf16* repo. The naive
  check reported a `model.vision_tower` "leak" on the 31B and 31 "leaks" on the 26B MoE (30 routers + 1
  vision) on correct builds. Note `layer_scalar` cannot trip it — it ends `_scalar`.
* **The verifier can silently no-op.** It keyed on `model.safetensors.index.json`; the
  `export_safetensors` path writes none, so for the Gemma-4 MoE build it printed
  `no index.json — cannot verify` and returned cleanly. Glob `*.safetensors` instead.
* **A raised failure can still be swallowed.** The runner wrapped the driver in a shell pipeline, so the
  `if` tested `sed`'s exit status: the 31B logged `VERIFY FAIL` and the runner printed ✅ on the next
  line. Make failures raise **and** do not lose the exit status in a pipe.

Current verification state of the six builds: three (Ornith, Glimmer, Mistral) passed the driver's own
check at build time; the 31B's logged verify was the false positive above and was **never re-run by the
driver**; the 26B MoE was **never checked by the driver at all**. Both were re-verified afterwards by
direct safetensors-header inspection (31B: 180 modules, 0 leaks; 26B: 11,610 modules, 0 leaks). All six
are structurally verified; five have never been loaded.

### Quark MXFP4 builds

Sizes are decimal GB of the artifact directory (GNU `du -h` rounds up, so it prints one unit higher).
Artifacts are bind-mounted read-only into the serving container as `/quant/<name>`.

| Artifact | Base | Arch | Quantized modules | Size (from base) | `exclude` | Status |
|---|---|---|---|---|---|---|
| `Qwen3.8-27B-MXFP4-Quark-RDNA4` | `Qwen/Qwen3.8-27B` (Apache-2.0) | `qwen3_5`, **dense** hybrid VL, 64 layers = 48 GDN linear-attn + 16 full-attn | **432** = 192 dense MLP (64×3) + 240 linear-attn projections (48×5) | 22.3 GB (20.7 GiB) from 55.6 GB — 40.1 % | 231 | **served + benchmarked** |
| `Ornith-1.0-35B-MXFP4-Quark-RDNA4` | `ornith-ai/Ornith-1.0-35B` (MIT per card; the old `deepreinforce-ai/Ornith-1.0-35B` id now redirects here) + MTP head grafted from `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` (Apache-2.0) | `qwen3_5_moe` VL, 40 layers = 30 GDN + 10 full-attn, 256 experts + 1 shared | **30,990** = 30,720 expert (40×256×3) + 120 shared-expert + 150 linear-attn | 23.0 GB (21.4 GiB) from 70.2 GB — 32.7 % | 1046 | structurally verified, **not yet loaded** |
| `Muse-Glimmer-30B-MXFP4-Quark-RDNA4` | `meta-models/Muse-Glimmer-30B` (Apache-2.0 + usage policy) | `muse_glimmer`, dense VL, 52 layers, GQA-2, 39 sliding (window 2048) + 13 full | **156** = 52×3 MLP | 29.1 GB (27.1 GiB) | 564 | structurally verified, **not yet loaded** |
| `Mistral-Small-3.2-24B-MXFP4-Quark-RDNA4` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` (Apache-2.0) | `mistral3`, dense, 40 layers + Pixtral-style 24-layer vision tower | **120** = 40×3 MLP | 18.5 GB (17.2 GiB) | 333 | structurally verified, **not yet loaded**; processor config missing (see quirks) |
| `gemma-4-31B-it-MXFP4-Quark-RDNA4` | `google/gemma-4-31B-it` | `gemma4`, dense, 60 layers = 50 sliding + 10 full, 27-layer vision tower | **180** = 60×3 MLP | 32.0 GB (29.8 GiB) from 62.5 GB — 1.96× | 419 | structurally verified, **not yet loaded** |
| `gemma-4-26B-A4B-it-MXFP4-Quark-RDNA4-moe` | `google/gemma-4-26B-A4B-it` | `gemma4` MoE, 30 layers × (128 experts + a dense MLP), 27-layer vision tower | **11,610** = 11,520 expert (128×30×3) + 90 dense MLP | 17.3 GB (16.1 GiB) from 51.6 GB — 2.98× | 337 | structurally verified, **not yet loaded**; single shard, **no index.json** |

Sizes above are the whole artifact directory except for the 27B row, which quotes tensor bytes from the
index (`du -sh` prints 21G). Base sizes are upstream bf16 index totals where a comparison is given.

## What's in the port
- `vllm/model_executor/kernels/linear/mxfp4/rdna.py` — `RdnaMxfp4LinearKernel`: weight-only (A16)
  MXFP4 dense linear for RDNA4 (no hardware MX datapath): in-kernel Triton dequant of 4-bit tiles
  to the activation dtype, then WMMA v2. Weights never materialize in high precision. Selected ahead
  of emulation on ROCm; also accepts W4A4 (Quark) configs by ignoring the activation quant.
- `vllm/model_executor/kernels/linear/mxfp4/rdna_fp8.py` — `RdnaMxfp4Fp8LinearKernel` (rc6, 2026-08-15):
  MXFP4 weights × e4m3 activations on RDNA4's **FP8 WMMA**. Decode (M ≤ 128): fused Triton GEMM that
  builds each E2M1 nibble as its exact e4m3 bit pattern, `tl.dot` per 32-wide K block, E8M0 block scale
  applied to the fp32 partial after the dot, per-(token, 32-K-group) dynamic e4m3 activations. Mid M
  (128–512): the weight-only in-kernel path. Prefill (M > 512): exact integer dequant to bf16 into a
  reused scratch + hipBLASLt bf16 GEMM. Selected ahead of the bf16-unpack kernel; `VLLM_RDNA_MXFP4_FP8=0`
  disables; `VLLM_RDNA_MXFP4_FP8_SKIP=<prefix,…>` keeps named layers on bf16 activations.
  Effect on Qwen3.8-27B MXFP4 TP2 (2× R9700), **all `--think raw`** — the bench's `/v1/completions` path at **temperature 0**, i.e. greedy. That is NOT the same as `--think off`, which uses `/v1/chat/completions` at temp 0.6 / top_p 0.95. The distinction matters here because this build runs MTP-3 speculative decoding: greedy decoding accepts most drafted tokens (mean acceptance 3.10), while sampled decoding rejects far more, so raw reads ≈61 tok/s where chat-off reads ≈51 on the same weights. Production sampling is the chat path; quote raw figures only against other raw figures: single-stream 51 → **61 tok/s**; short sweep c1 57 (= stock
  FP8), c32 aggregate 649 (old 600, FP8 430); 6k-prefill c8 29 (old 22, FP8 32). **Think-ON is a different, slower
  shape — same box, 2026-08-16: c1 46.5, c16 384, c32 531; do not compare think-OFF and think-ON numbers.**
  *(Bench-v4 update, 2026-08-24: the bench's raw essay path now samples at temp 0.7 / top_p 0.95
  over rotating distinct topics with a per-invocation nonce — replay-proof, nothing is ever
  regenerated — so v4 raw figures sit near the production sampling regime rather than greedy. The
  pre-v4 figures in this bullet, raw and think-ON alike, are replay-inflated and comparable only
  among themselves — not to v4 raw and not to any chat figure. See the "Measurement update
  2026-08-24" block below.)*
  gsm8k n=50 ×3 seeds and a 166-case private application regression suite unchanged vs the old kernel.
  **Scope: every think-OFF figure in this bullet was measured on the earlier RTN MXFP4 build of the same
  model, not on the Quark build now serving** — no think-OFF sweep exists for the Quark build. The two
  builds gate within noise on think-ON sweeps, so carrying the numbers over is an inference, not a
  measurement. Design lineage: Rob's `_matmul_fp8_ogs`
  (0.24 line, W8A8) — same per-K-group scale fold on WMMA v2; this one adds the MXFP4 unpack.
- Triton unified-attention: allow the 3D split-KV path for small-q spec-decode verify
  (`MAX_QLEN_3D=8`) — restores MTP/DFlash/DSpark verify throughput on gfx1201
  (branch `fix/unified-attn-3d-smallq` carries this alone).
- ROCm/gfx1201 build + container recipe (image below), FP8 via native FP8 WMMA.

## Release rc9 (2026-08-19) — the serving-performance batch, and the two-config A/B

Everything below is measured on 2× Radeon AI PRO R9700 (PCIe, TP2) serving Qwen3.8-27B at the full
native 262,144-token window with MTP-3. Numbers are single-run cells from a fixed harness
(**[Capicua25x/modelbench](https://github.com/Capicua25x/modelbench)** — `concurrency-bench.sh`, think ON,
`max_tokens 256`); accuracy gates are paired items at on-spec sampling, same repo.

**In the image/source (rc9 = rc8 + three attention-path changes, all env-gated):**
* **Hardware fp8 converts on gfx12 without rebuilding Triton** — Triton 3.6.0 open-codes `f32↔e4m3fn`
  (≈33 VALU + 10 `s_wait_alu` per element) although gfx1201 has `v_cvt_pk_fp8_f32`. A plain-text LLVM-IR
  extern library (`vllm/v1/attention/ops/rdnacvt.ll` + `rdna_cvt.py`, `tl.extern_elementwise`) emits the
  hardware ops and returns real fp8 tensors: fp8-Q attention kernel −33 %, inner loop 1,568 → 814
  instructions/iteration (= parity with the bf16-KV kernel), bit-identical outputs. Inert on bf16 KV.
  `VLLM_RDNA_HW_FP8CVT=0` for A/B.
* **`VLLM_RDNA_P_SCALE` (default 256)** — the software e4m3 downcast flushes softmax probabilities
  ≤ 2⁻¹⁰ of the row max; a 2⁸ exponent shift (undone in the epilogue) lowers the floor to 2⁻¹⁸.
  GSM8K strict on the fp8-KV config: 0.70 → 0.88. Inert on bf16 KV; `=1` restores upstream numerics.
* `VLLM_RDNA_TILE_PREFILL` — measurement knob for the fp8-Q 2D kv-tile (default 32 = upstream).

**Deployment findings you can use with ANY build (no code needed):**
* **`NCCL_PROTO=Simple`** for TP2 over PCIe on this pair: RCCL picks the LL protocol for the ≈640 KB
  decode all-reduces and LL is 2.8× slower than Simple here (205 vs 73 µs/op). Served effect on the
  FP8+fp8KV config: short-prompt c16 +17 %. Numerically identical. (Independently rediscovered by
  r/LocalLLaMA as a deadlock workaround — same flag, same hardware.)
* **`--mamba-ssm-cache-dtype bfloat16` is a CAPACITY lever for hybrid GDN models with spec-decode** —
  in `align` mode every request reserves `groups × (2 + num_speculative_tokens)` SSM-state pages
  regardless of length (15 pages/request here). On a page-poor config it is the difference between 22
  and 32 actually-running requests (fair-cell c32 aggregate 228 → 319 tok/s). On a page-rich config it
  buys little and can cost single-stream speed — measure per config; gate accuracy (the checkpoint asks
  for fp32 state; our gates: GSM8K ×3 and a 100-item long-context-reasoning set stayed in band).
* **Tuned per-shape w8a8 block-fp8 GEMM configs are worth +7-18% end-to-end on gfx1201** — vLLM ships
  no `AMD_Radeon_R9700` entries in `quantization/utils/configs/`, so every dense fp8 GEMM runs on
  default tile heuristics. Community-tuned configs (credit: **andysalerno** and **prcoe1**, whose
  [r9700-serving](https://github.com/andysalerno/R9700-serving) /
  [fork](https://github.com/prcoe1/r9700-serving) benchmarks surfaced the lever and its magnitude)
  measured on our serve: c1 +7%, c16 +18%, c32 +13%, accuracy gates unchanged (GSM8K, 166-test
  regression suite). Their repos carry no license, so their JSONs are not redistributed here; we will
  publish our own configs regenerated with vLLM's tuner (`benchmarks/kernels/benchmark_moe.py --tune`
  and the w8a8 equivalent) in a future release.
* **Do NOT enable vLLM's custom all-reduce on gfx12/PCIe** (`use_custom_allreduce` is vendor-gated to
  gfx94/95 for a reason): it initializes and returns garbage on this pair.

**The two-config A/B we run in production (both pass the same gates; pick by workload):**

| | **B · FP8 stock + fp8 KV** | **C · MXFP4 (Quark) + bf16 KV** |
|---|---|---|
| checkpoint | `Qwen/Qwen3.8-27B-FP8` | [`Capicua25x/Qwen3.8-27B-MXFP4-Quark-RDNA4`](https://huggingface.co/Capicua25x/Qwen3.8-27B-MXFP4-Quark-RDNA4) |
| extra flags | `--kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16` + `-e NCCL_PROTO=Simple` | `-e NCCL_PROTO=Simple` optional (neutral here) |
| KV pool @32 slots | **539k tokens (2.06× window)** | 415k (1.6×) |
| pick when | max context capacity / short-prompt throughput | best per-user speed at real context sizes |

*B and C both measured on the rc10 image (2026-08-20; B carries the in-tree tuned R9700 GEMM configs).*

**Throughput** (think ON, `max_tokens 256`, per-user / aggregate tok/s, warm serve, first-run cells).
*(Pre-v4, replay-era — absolute cells are inflated by the fixed-prompt replay confound (magnitude is shape-dependent — see the measurement-update block below); the
B-vs-C comparisons within these tables were measured like-for-like and stand as relative claims.
See the 2026-08-24 measurement update below for honest absolutes.)*
Short prompts (≈30 tok):

| users | B | C |
|---|---|---|
| 1 | 66 / 66 | 62 / 62 |
| 4 | 60 / 232 | 53 / 208 |
| 8 | 54 / 421 | 42 / 335 |
| 16 | 45 / 687 | 32 / 505 |
| 32 | 33 / **1,014** | 24 / 713 |
| 64 | 24 / **1,016** | 17 / 714 |

6,000-token prompts (RAG / long-system-prompt workloads) *(pre-v4, replay-era — see note above)*:

| users | B | C |
|---|---|---|
| 1 | 61 / 61 | 54 / 54 |
| 4 | 43 / 170 | 43 / 163 |
| 8 | 31 / 241 | 30 / 231 |
| 16 | 20 / 311 | 20 / 312 |
| 32 | 11 / 356 | 12 / 368 |

**Measurement update 2026-08-24 — bench v4 (honest absolutes).** Root cause: the v3 essay workload
reused **one fixed prompt**, and on a spec-decode server the stateful drafter partially **replays
previously generated text**, so acceptance and tok/s inflate with the server's own content history
(prose acceptance is also strongly topic-dependent).
[modelbench](https://github.com/Capicua25x/modelbench) v4 (commits `0619644` + `a04d4e6`) fixes it:
rotating distinct topics + a per-invocation nonce + temp 0.7 / top_p 0.95 — nothing is ever
regenerated — plus per-cell accepted/draft and tok/step columns scraped from `/metrics`. The
absolute essay cells above are replay-inflated (shape-dependent on re-measurement: ≈10-20% on most greedy-raw cells, up to ≈2x on the worst (a GB10-pair short cell), while one sampled-path 6k cell even measured slightly higher under v4 (different sampling provenance + build drift)); the relative B-vs-C
and rc-ladder deltas were like-for-like and stand. Honest v4 figures, config B (FP8 + MTP-3, rc10,
2× R9700 TP2, idle-verified; per-user / aggregate tok/s):

| workload (v4) | c1 | c4 | c16 | c32 | c64 |
|---|---|---|---|---|---|
| short essay | 59.1 (acc 1.52/3) | 49.5 / 183 | 37.1 / 541 | 27.0 / **789** | 20.1 / 825 |
| 6k-prefix essay | 55.4 (acc 1.60/3) | 44.1 / 168 | 21.8 / 327 | 12.8 / 390 | 8.4 / 397 |
| trivial (count-to-300) | 98.0 | — | 64.5 / 1,030 | 47.7 / 1,523 | 35.7 / 1,521 |

The 6k aggregate saturates around 390–400 from c=32 (c48: 10.0 / 373). Acceptance on novel prose is
a uniform 1.72–1.84 of 3 drafted in the multi-user 6k cells, vs a flat 2.99/3 on trivial — the
`--trivial` workload was always replay-free and its numbers were honest before and after v4.
Comfort ceiling at 20 tok/s/user: ≈16 in-flight at 6k prompts, ≈64 at short.

Config C (MXFP4 @ bf16 KV, same rc10 image, TP2, MTP-3, measured the same night, idle-verified):

| workload (v4) | c1 | c4 | c16 | c32 | c64 |
|---|---|---|---|---|---|
| short essay | 43.4 (acc 1.14/3) | 43.7 / 166 | 30.2 / 448 | 20.5 / 602 | 13.8 / 576 |
| 6k-prefix essay | 54.1 (acc 1.90/3) | 40.9 / 158 | 21.5 / 325 | 12.6 / 387 | 8.7 / 395 |
| trivial (count-to-300) | 64.5 | 74.8 / 299 | 52.6 / 741 | 36.1 / 1,042 | 27.1 / 1,039 |

**The C-vs-B story under honest measurement: at realistic ≈6k context the two arms are equal within
noise at every level** (327/390/397 vs 325/387/395 aggregate; same ≈16-user comfort ceiling). The
FP8 arm's edge is confined to compute-bound shapes — trivial peak 1,523 vs 1,042 (≈1.46×) and short
prompts (≈1.3–1.4×) at equal speculative acceptance, i.e. MXFP4 dequant cost, not drafting. What C
buys for that cost: the full 262k native window on the same silicon. Choose by workload: burst
short-query capacity → B; maximum context → C; at real context sizes the throughput trade is nil.

## Single card (1× R9700 / 32GB-class)

**Single-card path: MXFP4 @ fp8 KV** — the only quant that fits (weights ≈21 GB); fp8 KV stretches
the window from 32k to 104k. It serves — with real limits: ≈4 concurrent users and a window still
well short of the 262k the multi-card port gets on the same silicon. llama.cpp/GGUF can also run
this model on one card for plain chat, but skips the batching, prefix caching, and window this
config gets from the same card — not the recommendation unless that's genuinely all you need.

Measured 2026-08-25 on 1× R9700, rc10 image, bench v4, KV pool 113,642 tokens (127,348 at 4 slots)
→ 104k-token window:

| workload (v4, TP1 + fp8 KV, 104k window, 4 slots) | c1 | c2 | c4 |
|---|---|---|---|
| trivial | 43.8 / 44 | 40.7 / 81 | 39.9 / 143 |
| short essay | 29.0 / 29 | 27.6 / 51 | 23.5 / 85 |
| 6k-prefix essay | 32.4 / 32 | 27.0 / 51 | 23.7 / 90 |

Stays ≥20 tok/s/user through c4 (untested beyond). Serve command — one device pair, TP1, 104k
window, 4 slots:

```bash
docker run --rm --name vllm-qwen --network=host \
  --device=/dev/kfd --device=/dev/dri/renderD128 --device=/dev/dri/card1 \
  --group-add=video --group-add=render --ipc=host \
  -v /path/to/quants:/quant:ro -e HF_HUB_OFFLINE=1 \
  --entrypoint /usr/local/bin/vllm capicua25x/vllm-rocm-rdna4:latest \
  serve /quant/Qwen3.8-27B-MXFP4-Quark-RDNA4 \
  --served-model-name qwen --port 8011 --trust-remote-code \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 106496 \
  --attention-backend TRITON_ATTN --enable-prefix-caching \
  --max-num-seqs 4 --max-num-batched-tokens 8000 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 --max-cudagraph-capture-size 32 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
```

**16GB cards (RX 9070 XT): this 27B does not fit — in ANY runtime.** Our quants need 21 GB of
weights; even a Q4 GGUF (≈15.5 GB) technically loads and then leaves no room for KV, i.e. no
usable context window. llama.cpp's 16GB options are CPU offload (at a large speed cost) or a
smaller model — the latter is the honest recommendation. The port's kernels run fine on gfx1200
silicon; pair the card with a model whose weights + context leave real headroom.

Quality note: fp8 KV over this MXFP4 quant carries one accuracy smoke (gsm8k n=50 thinking-on:
0.94 flexible / 0.88 strict — within the ±0.04 sampling noise of the bf16-KV baseline's 0.91–0.93,
at its lower edge). Validate on your own workload before committing; a KV-calibrated variant with
scale side-files exists if deeper validation matters to you.


**Full serve commands** (2× R9700 shown; adjust `--device` paths to your cards; TP2, full native 262k
window, MTP-3, 32 slots). The A/B pair we run in production:

```bash
# Quick start B · FP8 @ fp8 KV — max context capacity (KV pool ≈ 2× the 262k window)
docker run --rm --name vllm-qwen --network=host \
  --device=/dev/kfd --device=/dev/dri/renderD128 --device=/dev/dri/renderD129 \
  --group-add=video --group-add=render --ipc=host \
  -e NCCL_PROTO=Simple \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /usr/local/bin/vllm capicua25x/vllm-rocm-rdna4:0.26.1-rdna4-rc10 \
  serve Qwen/Qwen3.8-27B-FP8 --served-model-name qwen --port 8011 --trust-remote-code \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 262144 \
  --attention-backend TRITON_ATTN --enable-prefix-caching \
  --max-num-seqs 32 --max-num-batched-tokens 8000 --max-cudagraph-capture-size 128 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'

# Quick start C · MXFP4 @ bf16 KV — max multi-user throughput at real context sizes
#   = the same command with:  serve Capicua25x/Qwen3.8-27B-MXFP4-Quark-RDNA4
#     and WITHOUT:            --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16
```

rc9's env knobs (`VLLM_RDNA_HW_FP8CVT=1`, `VLLM_RDNA_P_SCALE=256`) are the defaults — active on B's
fp8-KV path, inert on C's bf16-KV path; set to `0`/`1` respectively only to A/B against upstream
behavior. On B, the checkpoint ships no KV-cache scale tensors (scale 1.0; our amax survey peaked at
147 vs e4m3's 448, so nothing clips).

So far Qwen3.8-27B behaves correctly on both A/B configurations under production traffic and the
gates above; we will fold the verdict back here when the A/B concludes.

## Image (Docker Hub)
`capicua25x/vllm-rocm-rdna4:0.26.1-rdna4-rc10` (= `:0.26.1-rdna4` = `:latest` after the 2026-08-19
gates) — rc9 = rc8 + hardware fp8 converts + `VLLM_RDNA_P_SCALE` + the prefill-tile knob (all inert on
bf16 KV). rc8 = rc7 + the re-tuned MXFP4 tile table (`sha256:1fffe1cb…`). rc6 = rc5 +
`RdnaMxfp4Fp8LinearKernel`; rc5 = `sha256:0f5cbc40…` (also tagged `glimmer-qwen38-rc5`, kept).
Previous generation: `:0.19.1`.

## Models on this port

Current profile: **Qwen3.8-27B** (dense hybrid GDN/attention, VL, native MTP) — FP8 stock and our
Quark MXFP4 build, both accuracy-gated and in the A/B above. More model profiles (35B-A3B MoE,
Gemma-4, Mistral) will be offered later; five structurally-verified Quark builds remain unserved
and unreleased.

## Accuracy (AA class-A, paired items, seed 1234, on-spec sampling)

**ref** = the same checkpoint served in bf16 by a cloud provider (our parity reference). Same judge
for all judged rows. Cells show the most recent run at the stated n on the shipping config; ±2 items
is the noise band.

| benchmark (n) | ref (bf16) | B | C |
|---|---|---|---|
| GSM8K think, flex·strict (50) | 0.96·0.82 | 0.98·0.86 | 0.98·0.96 |
| — seeds 7 / 99 | — | 0.96·0.84 / 0.96·0.90 | 0.98·0.98 / 0.96·0.90 |
| GSM8K nothink (50) | 0.98·0.98 | 0.98·0.98 | 0.98·0.98 |
| IFEval inst·prompt (80) | 0.97·0.95 | 0.98·0.98 | 0.95·0.91 |
| AA-LCR judged (100) | 0.78 | 0.77 · 0.81 (s1234·s99) | 0.78 |
| GPQA-Diamond (60) | 0.78 | 0.85 | 0.92 |
| AIME'25 (30) | 0.93 | 0.97 | 0.93 |
| τ²-telecom (114) | 0.939 | 0.939 | 0.904 |
| τ²-airline (50) | 0.760 | 0.860 | 0.840 |
| τ²-retail (60) | 0.850 | 0.817 | 0.767 |
| MMLU-Pro (1120) | 0.804 | 0.817 | ⏳ |
| Terminal-Bench hard (44, 3600s/task) | 0.273 | 0.341 | ⏳ |

Reading caveats, condensed: GSM8K under continuous batching is seed-labelled but not deterministic
(treat single cells as draws from the seed spread); the judged rows flip ≈1 item/100 on re-judge —
never read a 1–2 item gap as a quantization result; the τ² reference is a repaired number
(12 provider-side sim deaths re-run) while C's 0.904 is unrepaired. B's τ²-telecom is a
from-scratch 2026-08-21 run (107/114, think ON, 114/114 user_stop, 0 censored) — reference
parity. Earlier B cells near 0.63 came from a launcher that silently dropped thinking; those
are withdrawn, and `think_preflight.sh` now guards every local τ² launch. τ²-airline and τ²-retail are the same
08-21 think-ON batch as telecom (50/50 and 60/60 user_stop). Terminal-Bench is
terminal-bench-core with the terminus-2 agent over a 44-task subset — NOT Terminal-Bench 2.1,
so it is not comparable to vendor-card figures; both cells are the full 44 with no exclusions.
C's MMLU-Pro and Terminal-Bench are not yet measured.

**Weights release:** the 27B Quark MXFP4 build is published:
[`Capicua25x/Qwen3.8-27B-MXFP4-Quark-RDNA4`](https://huggingface.co/Capicua25x/Qwen3.8-27B-MXFP4-Quark-RDNA4)
(model card carries the base licence, attribution, statement of modification, and the measured
accuracy/throughput tables). The five unserved builds remain unreleased — some have open base-licence
questions and none has serving evidence.

## Not here
No PRs upstream by choice — the branch is carried on this fork (patches are separable:
`feat/rdna4-mxfp4-linear`, `fix/unified-attn-3d-smallq`).
