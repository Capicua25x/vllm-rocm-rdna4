# vllm-rocm-rdna4 — vLLM 0.19.1 on AMD RDNA4 (gfx1201): MXFP4 (Qwen) + NVFP4 (Gemma-4)

Build recipe for **vLLM 0.19.1** with **RDNA4 (gfx1201 / Radeon AI PRO R9700) enablement** on ROCm,
serving compressed-tensors quantized MoE models — **MXFP4** (e.g. Qwen3.5/3.6-MoE) and **NVFP4** (Gemma-4).

> ## 🙏 Built on Rob Smith's RDNA4 work
> This recipe **forward-ports the gfx1201 kernel enablement that [Rob Smith](https://hub.docker.com/r/tcclaviger/vllm-rocm-mxfp4-nvfp4)**
> (`tcclaviger`, image [`tcclaviger/vllm-rocm-mxfp4-nvfp4`](https://hub.docker.com/r/tcclaviger/vllm-rocm-mxfp4-nvfp4))
> cracked for vLLM 0.18.1. His working RDNA4 base — complete gfx1201 hipBLASLt 1.2 plus the MXFP4/NVFP4
> MoE kernels — is the foundation this entire repo stands on. **Without Rob's work, none of this exists.**
> Everything here is a forward-port and extension of what he built first. (Full attribution
> [below](#attribution--license).)

> **Why this build exists (the headline use case):** to bring up **Gemma-4 (NVFP4 MoE) as a
> first-class RDNA4 model** — a second, independent model family on the *same* RDNA4 vLLM image as the
> Qwen MXFP4 path. That was the hard part, and it drove the novel work: a new **head-512 flash-prefill
> kernel** (Gemma's full-attention layers are head_dim-512), an **fp8-KV** fix for fast cold 256k, and a
> **`gemma4_assistant` draft-model spec-decode backported from vLLM 0.23.0** — all under
> [`gemma-nvfp4/`](gemma-nvfp4/). What's original (the kernel, the MoE fix) vs. borrowed (llama.cpp's
> chunking blueprint) vs. backported (the 0.23.0 draft) is spelled out in **Attribution** below.
>
> **Secondary payoff — also helps the Qwen3.6-35B-A3B MXFP4 distill, not just Gemma.** Two serving
> tweaks found along the way: (1) `--attention-backend TRITON_ATTN` recovers **6K-prompt concurrency**
> (≈5.1 → 19.4 tok/s at 32 users — table below); (2) the MTP **3D spec-verify gate**
> (`VLLM_TRITON_3D_SMALL_QLEN`) is a **single-stream 6K-latency win — +15% on the Qwen distill**
> (compiled, 80.8 → 93.3 tok/s) and **+114% on Gemma** (32.5 → 69.5), output-lossless, neutral at high
> concurrency, no downside (reproduce via `gemma-nvfp4/port-0191/launch-qwen-test.sh`).

> **Validated:** Qwen3.6-35B-A3B DeepSeek-V4-Pro-Thinking distill (MXFP4 + vision) — passes a downstream
> task-accuracy suite (NDA, withheld), ~109 tok/s single-stream (MTP-3), cudagraph-safe. **Gemma-4-26B-A4B
> NVFP4** — GPU-confirmed; admits 256k, largest *measured* cold prefill 236k = 346 s (262k extrapolates
> ~390 s) — under the 600 s bar (details in [`gemma-nvfp4/`](gemma-nvfp4/)).

## Tested & validated (3-model battery, 2026-06-27)

**Stack:** 2× AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB ea.), tensor-parallel 2; vLLM 0.19.1 on
ROCm (image `capicua25x/vllm-rocm-rdna4:0.19.1`), `--attention-backend TRITON_ATTN`, MTP-3 spec-decode
(output-lossless), compiled decode, `--max-model-len 262144 --gpu-memory-utilization 0.92
--max-num-seqs 64 --enable-prefix-caching`.

Three models on the **same** image — Qwen3.6-35B-A3B (base), the DSV4Pro distill, and Gemma-4-26B-A4B
NVFP4 — compared apples-to-apples. Full tables:
[`gemma-nvfp4/port-0191/multimodel-bench/MULTIMODEL-BENCH.md`](gemma-nvfp4/port-0191/multimodel-bench/MULTIMODEL-BENCH.md).

- **Throughput / concurrency bench** — single-stream decode **100–109 tok/s** (all three; Gemma matches
  Qwen). Short-prompt concurrency @16 favours Qwen (~760 agg vs Gemma 226); 6k prefill-heavy @32/64
  **inverts** — Gemma **635 / 816** agg, ahead of Qwen (≤591 / ≤704).
- **Cold max-context prefill** (~205–212k, cache-miss) — Qwen ~2045–2068 tok/s (~102–104 s); Gemma
  ~642 tok/s (**320 s** — ~3× slower, but **under the 600 s bar** thanks to the head-512 flash kernel).
  (Largest *measured* Gemma cold point is 236k = 346 s; see `BLOCKER2-fp8-capacity-RESOLVED.md`.)
- **Compaction survival + performance** (~146–154k round-trip: prefill → summarize → continue) — Qwen
  **69.4 / 65.5 s**, Gemma **220.3 s**; all complete, recall correctly, **well under 600 s**.
- **Regression testing** — a 137-case downstream task-accuracy suite (NDA, withheld) through each model:
  base **134·3W·0F**, distill **135·2W·0F**, Gemma **133·4W·0F** → **zero hard failures**, Gemma within 2
  of the tuned distill even on Qwen-shaped prompts (the cross-family robustness result).
- **Vision** — all three describe the test image correctly; Gemma fastest (**0.5 s** vs 3.3–3.4 s).
- **Audio — UNTESTED.** The bench leg was skipped; this recipe neither wires nor validates an audio input
  path (Qwen variants are text+vision; Gemma-4's audio capability, if any, is out of scope here).

*Method caveats (per the bench doc): one run per cell; the regression suite is non-deterministic
(`top_p=0.95`, so ±1–2 on the scorecard is noise); KV dtype is each model's best-fit (Gemma fp8 for 256k
admission, Qwen native).*

## Run it (Docker — bring your own model)

The image is the **engine** (vLLM + ROCm + the RDNA4 kernels). **No weights are baked** — you bind-mount
your own model from your Hugging Face cache, the same way the upstream RDNA4 images work.

```bash
docker pull capicua25x/vllm-rocm-rdna4:0.19.1   # or :latest
```

### Models validated on this image (the exact ones benchmarked)

| Model | Quant | Hugging Face repo | Notes |
|---|---|---|---|
| Qwen3.6-35B-A3B (base) | MXFP4 | [`pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`](https://huggingface.co/pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4) | stock Qwen MoE |
| Qwen3.6-35B-A3B DSV4Pro distill | MXFP4 + vision | [`Capicua25x/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4-Vision`](https://huggingface.co/Capicua25x/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4-Vision) | thinking distill; needs `--trust-remote-code` |
| Gemma-4-26B-A4B-it | NVFP4 | [`RedHatAI/gemma-4-26B-A4B-it-NVFP4`](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-NVFP4) | + draft [`google/gemma-4-26B-A4B-it-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant) for MTP spec-decode |

Any other compressed-tensors **MXFP4** or **NVFP4** MoE should work — these are just the ones with measured
numbers (see [`MULTIMODEL-BENCH.md`](gemma-nvfp4/port-0191/multimodel-bench/MULTIMODEL-BENCH.md)). Fetch a
model into your HF cache first, e.g.:

```bash
huggingface-cli download RedHatAI/gemma-4-26B-A4B-it-NVFP4
```

### Serve — Qwen (MXFP4)

```bash
docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render --ipc=host \
  --security-opt=no-new-privileges --cap-drop=ALL --cap-add=DAC_READ_SEARCH --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e HF_HUB_OFFLINE=1 -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  capicua25x/vllm-rocm-rdna4:0.19.1 \
  --model pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4 \
  --served-model-name qwen --port 8011 --trust-remote-code \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 262144 \
  --attention-backend TRITON_ATTN --enable-prefix-caching --max-num-seqs 64 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

(For the vision distill, point `--model` at the `Capicua25x/...` repo and add
`--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`.)

### Serve — Gemma-4 (NVFP4)

```bash
docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render --ipc=host \
  --security-opt=no-new-privileges --cap-drop=ALL --cap-add=DAC_READ_SEARCH --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e HF_HUB_OFFLINE=1 -e VLLM_FA_HEADCHUNK=64 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  capicua25x/vllm-rocm-rdna4:0.19.1 \
  --model RedHatAI/gemma-4-26B-A4B-it-NVFP4 \
  --served-model-name gemma --port 8011 --trust-remote-code \
  --attention-backend TRITON_ATTN --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 262144 \
  --enable-prefix-caching --max-num-seqs 64 --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"model":"google/gemma-4-26B-A4B-it-assistant","method":"mtp","num_speculative_tokens":3}'
```

**Gemma flags that matter:** `VLLM_FA_HEADCHUNK=64` (head-512 flash kernel → fast cold prefill),
`--kv-cache-dtype fp8` (admits cold 256k), the `gemma4_assistant` draft for MTP, and
`cudagraph_mode=FULL_DECODE_ONLY` (compiled decode + eager prefill). The *why* is in
[`gemma-nvfp4/`](gemma-nvfp4/).

> **`--attention-backend TRITON_ATTN` is load-bearing on 0.19.1** — the default `ROCM_ATTN` collapses
> spec-decode throughput at concurrency on gfx1201. Full numbers in **Serve** below.

Device mounts use the generic `--device=/dev/kfd --device=/dev/dri` (works on any host); to pin specific
GPUs, set `HIP_VISIBLE_DEVICES` on the container.

## What this is (and isn't)

The AMD *official* `rocm/vllm gfx120X` image can't run non-trivial MXFP4 models (its preview gfx1201
hipBLASLt is missing fused-bias GEMM kernels). Rob Smith's `tcclaviger/vllm-rocm-mxfp4-nvfp4` image
solved the RDNA4 kernel problem on **vLLM 0.18.1** but is frozen. This recipe **forward-ports the
RDNA4 enablement onto vLLM 0.19.1**, built *on Rob's working base* (complete gfx1201 hipBLASLt 1.2,
torch 2.10 / ROCm 7.2.1) — sidestepping the broken official-image GEMM library entirely.

It is a **build recipe**, not a redistribution of anyone's weights — no model weights are baked in
(they are bind-mounted at runtime from the HF cache).

### Why 0.19.1 (and not a newer vLLM)?

0.19.1 is the **last vLLM release that still runs on torch 2.10.** vLLM **0.20.0 bumped torch
2.10 → 2.11**, and every release since pins 2.11. The only known-working RDNA4 (gfx1201) base —
Rob's image — is built on **torch 2.10 / ROCm 7.2.1**, and no torch-2.11 gfx1201 base exists yet
(AMD's own official `gfx120X` image is likewise frozen at torch 2.10 / vLLM 0.19.1). So 0.19.1 is
the *newest* vLLM that runs on a working RDNA4 stack — a hard ceiling, not a conservative pick.

This was confirmed the hard way: a vanilla **0.23** build on torch 2.10 fails at the pip build
backend (it errors before the C-extension compile even starts), and several of the files this recipe
patches were refactored or removed by 0.23. Until a torch-2.11 gfx1201 base lands, 0.19.1 is the
target.

## Novel bit 1 — the monolithic-TRITON MoE OOB fix (makes Qwen MXFP4 run)

vLLM 0.19.1's new `oracle/mxfp4.py` backend selector routes RDNA4 SiLU MoEs to `TRITON_UNFUSED`,
whose `UnfusedOAITritonExperts.apply` does a **raw torch fancy-index gather** on the padded
256-expert dispatch space — which over-reads (`dst_indx` reaches 251 into a 224-row buffer) and
faults the GPU (`HSA_STATUS_ERROR_EXCEPTION 0x1016`) on real ragged routing. Fix = mirror Rob's
0.18.1 **monolithic** path (`patches/vllm-0.19.1-rdna4-mxfp4.patch`):
1. `triton_kernel_fused_experts`: add an `_is_rdna4` branch (GEMM1 `fused_activation=None` full-N →
   `torch.ops._C.silu_and_mul` → GEMM2 with `scatter_indx`) — gather/scatter stay inside the masked
   `matmul_ogs`, no raw torch gather → no OOB.
2. `OAITritonMxfp4ExpertsMonolithic._supports_activation`: accept `MoEActivation.SILU` on `on_gfx1x()`
   so the oracle picks the monolithic TRITON path.

## Novel bit 2 — the head-512 flash-prefill kernel (makes Gemma cold-prefill fast)

Gemma-4's full-attention layers are **head_dim-512**. vLLM's stock Triton prefill stages the entire
512-deep Q tile in LDS (`BLOCK_M 128 × 512 × 2B ≈ 128 KiB`), but RDNA4 has only ~64 KiB LDS/CU — so the
kernel can only launch at minimum tile size with ~1 resident wave-block: occupancy-starved, no latency
hiding. That is the ~3× cold-prefill gap. The fix (`VLLM_FA_HEADCHUNK=64`,
[`gemma-nvfp4/port-0191/head512-flash-prefill.patch`](gemma-nvfp4/port-0191/head512-flash-prefill.patch))
streams the 512-deep head dimension in **64-wide chunks** for *both* the QK and PV steps, so the staged
tile never exceeds `nbatch_fa × nbatch_K` (64×64) — the full head is never resident. LDS drops
**65,536 → 6,144 B (−91%)** (prototype kernel; the production kernel adds paged-KV / fp8-dequant /
mask buffers, so its absolute LDS is higher — the *speedup* is what's validated on the real kernel),
occupancy ~1 → ~10. Same-window A/B: **3.03× @63k, 3.50× @120k** cold
prefill. The online-softmax / masking math is **byte-identical** to stock; only the matmul operands'
memory layout changes. It is **default-OFF** (env-gated; with the var unset the stock kernel runs
byte-identical) and engages only on head-512 + RDNA4. Blueprint borrowed from llama.cpp's `nbatch_K`
chunking (credited below); correctness is pinned by a differential test (see FAQ → *Is it correct?*).
Full design + the WMMA dead-ends ruled out:
[`gemma-nvfp4/port-0191/cold-prefill-debug/FLASH_HEAD512_DESIGN.md`](gemma-nvfp4/port-0191/cold-prefill-debug/FLASH_HEAD512_DESIGN.md).

## Build

GPU-free (~3 min). Requires Rob's base image present locally.

```bash
docker build -t capicua25x/vllm-rocm-rdna4:0.19.1 -f Dockerfile .
```

### Gotchas (load-bearing, learned the hard way)

- **librccl symlink must precede the editable install.** cmake-configure's `roc::rccl` imported
  target references `librccl.so.1.0.70201`, which Rob's image lacks (actual lib is `.70101`). The
  Dockerfile symlinks it *before* `pip install -e`. Out of order → `Configuring incomplete` fast-fail.
- **Keep `.deps/triton_kernels-src/.git` in the build context.** cmake `FETCHCONTENT` needs it; a
  `**/.git` in `.dockerignore` silently drops it (context 1GB → 293MB) and configure fails at ~5s.
- **cmake/ninja live in `/app/.venv`**, but the uv-spawned build subprocess doesn't inherit that
  PATH — the Dockerfile symlinks them into `/usr/local/bin`.
- **Do NOT set `HIPBLASLT_TENSILE_LIBPATH`** — Rob's complete hipBLASLt 1.2 is the working default.
- `--no-deps` keeps Rob's torch 2.10 / ROCm 7.2.1 untouched.

## Serve

See `example-serve-wrapper.sh` (TP2 across 2× R9700, cudagraph on, MTP `num_speculative_tokens=3`,
`qwen3_xml` tool parser + `qwen3` reasoning parser, full hardening). ENTRYPOINT is the api_server
module; pass `--model <repo>` + args.

### ⚠ Load-bearing serve flag: `--attention-backend TRITON_ATTN` (confirmed 2026-06-24)

vLLM 0.19.1 made **`ROCM_ATTN` the default** attention backend, and ROCM_ATTN **collapses spec-decode
(MTP) throughput at concurrency on gfx1201**. Measured at 6K-token prompts:

| concurrency | ROCM_ATTN (0.19.1 default) | **TRITON_ATTN** | Rob's 0.18.1 image |
|---|---|---|---|
| 1 / 16 / 32 users | 61 / 12.5 / **5.1** tok/s | **92 / 25.7 / 19.4** | 82 / 26.5 / 20.9 |
| usable ceiling | ~1 user | **~16 users** | ~16 users |

`--attention-backend TRITON_ATTN` recovers concurrency to match the 0.18.1 image *and* beats its
single-stream by ~13%. **The `VLLM_ATTENTION_BACKEND` env var is NOT read in 0.19.1 — use the
`--attention-backend` CLI arg** (already set in `example-serve-wrapper.sh`). `ROCM_AITER_UNIFIED_ATTN`
is reportedly faster still but needs a newer AITER / CDNA-gate relax (`ImportError` on
`aiter.ops.triton.unified_attention` as of this image) — a future optimization, not required to ship.

## Attribution & license

This recipe assembles permissively-licensed components and credits them:
- **vLLM** — Apache-2.0 (https://github.com/vllm-project/vllm). Our edits (`patches/`) are derivative
  works under Apache-2.0.
- **Gemma-4 `gemma4_assistant` draft-model spec-decode** (`gemma-nvfp4/port-0191/*.patch.md`,
  `gemma4*.py`) — **backported from vLLM 0.23.0** (Apache-2.0) onto the 0.19.1 tree: 2 new model files
  (`gemma4.py` / `gemma4_mtp.py` proposer) + 5 framework patches (`eagle.py`, `speculative.py`,
  `gpu_model_runner.py`, `registry.py`, `model_arch_config_convertor.py`). Same upstream, newer
  revision than the 0.19.1 base; 0.23.0 won't build on this stack (torch wall), so the feature was
  taken from its source, not its binary.
- **RDNA4 MXFP4 kernel enablement** (`graft/tk/*`, the `rdna_value` dequant, the `_matmul_ogs` /
  `opt_flags` / `target_info` RDNA branches) — forward-ported from **Rob Smith's**
  `tcclaviger/vllm-rocm-mxfp4-nvfp4` RDNA4 work. Thanks to Rob for cracking the gfx1201 kernels.
- **Head-512 flash-prefill kernel design** — a clean-room Triton re-implementation of **llama.cpp's**
  (MIT) RDNA4 head-512 flash-attention recipe (`fattn-tile.cuh` `get_config_amd_rdna`: `nbatch_K=64`
  head-dim chunking + `nbatch_fa=64` KV tiling). No code copied — the tile-config/algorithm was the
  blueprint; llama.cpp's *measured* exclusion of WMMA at head-512 on RDNA4 also steered us off that
  dead end. Credit to that project.
- **Triton / triton_kernels** — MIT.
- **AMD ROCm** (hipBLASLt, rocBLAS, RCCL, HIP) — MIT.
- **Runtime base stack** (inherited via Rob's image, not modified here): **PyTorch** — BSD-3-Clause;
  **HuggingFace Transformers** — Apache-2.0.
- **Models** — not redistributed here (this is a recipe; weights are bind-mounted at runtime):
  **Qwen3.5/3.6-MoE** (Alibaba) and **Gemma-4** (Google), both **Apache-2.0**; the served NVFP4 build is
  **`RedHatAI/gemma-4-26B-A4B-it-NVFP4`** (RedHat AI's quantization), and the validated Qwen build is a
  DeepSeek-V4-Pro-Thinking distill lineage. Always confirm the license on the specific model card you
  deploy.

The Apache-2.0 license text is in [`LICENSE`](LICENSE); the change/attribution summary is in
[`NOTICE`](NOTICE).
