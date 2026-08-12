# gemma-nvfp4 — RDNA4 NVFP4 MoE + linear port for Gemma-4 (✅ GPU-CONFIRMED 2026-06-25)

NVFP4 companion to the **confirmed, in-production MXFP4 recipe** in the parent dir
(`../patches`, `../graft`, `../README.md` — Qwen3.6-35B-A3B MXFP4). This dir forward-ports the
**full NVFP4 path** (MoE **and** linear) so the same vLLM 0.19.1 / RDNA4 (gfx1201, 2× R9700) image
can also serve **`RedHatAI/gemma-4-26B-A4B-it-NVFP4`** (4B-active MoE — Qwen-class active params).

> **STATUS: ✅ GPU-CONFIRMED on 2× R9700 (gfx1201), 2026-06-25.** Loads + generates coherently,
> eager **and** torch.compile + CUDA-graph. First Gemma-4 NVFP4 MoE on RDNA4. This is the gate the
> parent README's public release was waiting on. The MXFP4 Qwen recipe is unaffected — additive.

## Results (2026-06-25, 2× R9700 TP2, `capicua25x/vllm-rocm-rdna4:0.19.1`)
| Check | Result |
|---|---|
| Load | ✅ eager **and** compiled (torch.compile 48s + cudagraph capture clean) |
| Coherence | ✅ self-IDs as Google/Gemma; fluent multilingual generation |
| Correctness | ✅ `17×23 = 391` (eager + compiled) — proves GAP-1 routing picks correct experts |
| Vision / multimodal | ✅ read "ACME 391" + identified the red circle |
| Single-stream | 62.4 tok/s eager / 65.9 compiled (short); ~49.5 @6K prompt — **beats 0.18.1's 45.3** |
| Concurrency @6K (1/16/32) | per-user 49.5 / 18.5 / 19.8 ; aggregate 50 / 295 / 424 tok/s — **matches Qwen at 32** (19.8≈19.4), graceful, no collapse |
| VRAM (TP2) | model 9.46 GiB; ~31 GB/GPU resident; KV ~18–19 GiB |

## Long-context: cold 256k works (fp8-KV)

With the locked config (`--kv-cache-dtype fp8` + `VLLM_FA_HEADCHUNK=64`), Gemma-4 **admits and
cold-prefills ≥256k under the 600s bar** on this stack. Measured (isolated, eager, TRITON_ATTN):

- cold needle **201k → 316s, recall 20/20**
- **cold prefill 236k → 346s (~682 tok/s)** — the clean cold ~256k headline
- over-length (>262144) → clean **HTTP 400** (no hang)
- 157k context-compaction completes + recalls (~3.4 min eager / ~5.5 min with MTP+draftwin)

> An earlier bring-up reported a hard "admission deadlock" at ~175k. Root-caused and **resolved**:
> it was simply the **fp16 KV-cache token capacity** (vLLM hung waiting for KV blocks instead of
> cleanly rejecting). `--kv-cache-dtype fp8` lifts capacity to **333k tokens > the 262k max-model-len**,
> making the deadlock structurally impossible. The "ceiling ~175k / does not match Qwen" notes in
> `AB_RESULT_3dprefill.md` are obsolete — see
> [`port-0191/cold-prefill-debug/BLOCKER2-fp8-capacity-RESOLVED.md`](port-0191/cold-prefill-debug/BLOCKER2-fp8-capacity-RESOLVED.md).

Honest remaining caveats (why Gemma is the **experimental** tier, not the dogfooded daily driver):

- Cold prefill is **~3× slower than Qwen** (still under the 600s compaction bar).
- **MTP roughly doubles cold prefill** (the draft also cold-prefills the full context) — for
  compaction use eager / no-spec or `GEMMA4_MTP_DRAFT_FULL_WINDOW=1024`. MTP helps steady-state *decode*.
- Short-prompt decode-bound concurrency is weaker than Qwen.
- Leans on the grafted NVFP4 path; benches are single-run.

## The fix has THREE parts (the staged port had only the first; the rest was found at bring-up)
1. **MoE routing** — `rdna_nvfp4_moe.py` GAP-1: allow `RoutingMethodType.Custom` (=6, Gemma's gating)
   and compute Gemma's exact routing (softmax-over-ALL → top-k → renormalize) in `apply()` instead of
   `fused_topk`. Reads `moe_config.routing_method` (0.19.1 renamed from `routing_method_type`).
2. **MoE backend + RDNA kernels** — `oracle/nvfp4.py` short-circuits gfx1x → `RDNA_TRITON` (before any
   FlashInfer/CUTLASS/Marlin import, which fail on ROCm) + `nvfp4_emulation_utils.py` RDNA dequant kernel.
3. **Linear scheme (NEW — the decisive missing half)** — `compressed_tensors_w4a4_nvfp4.py`: on gfx1x,
   set `backend=None`, keep weights packed (uint8) + linear scales, runtime-dequant NVFP4→bf16 via
   `invoke_nvfp4_linear_kernel`. Without this, stock 0.19.1 routes linear NVFP4 to **Marlin**
   (`gptq_marlin_repack` is **not compiled on ROCm** → `AttributeError`), and the emulation fallback
   assumes **swizzled** scales while RedHatAI stores them **linear** (→ reshape error). Ported from
   Rob's 0.18.1 scheme; **zero API drift** (3 hunks: `__init__`, `process_weights_after_loading`,
   `apply_weights`) — diff is in the patch.

## Working invocation (CRITICAL — the staged launch script was wrong on these)
- **Bind-mount/bake into `/build/vllm/vllm/...`, NOT `/app/vllm/...`.** The 0.19.1 image runs vLLM from
  the editable install at `/build/vllm`; `/app/vllm` is a stale Apr-3 tree. The 0.19.1 bake applied the
  MXFP4 graft but **never the NVFP4 graft** into `/build` — apply this patch + drop in the experts kernel
  + the linear scheme there (4 files total).
- **Pass the model as `--model <repo>` (a FLAG), not a positional `model_tag`.** The image ENTRYPOINT is
  `python -m vllm.entrypoints.openai.api_server`; with `HF_HUB_OFFLINE=1` a positional sets a bad
  `revision` → `LocalEntryNotFoundError`. (Matches the Qwen launch wrapper.)
- `--attention-backend TRITON_ATTN` (same as the Qwen migration — avoids the ROCM_ATTN concurrency cliff).
- `HF_HUB_OFFLINE=1` only — **no** `VLLM_USE_NVFP4_CT_EMULATIONS` (the scheme handles linear RDNA dequant).
- `--enforce-eager` is **optional**: torch.compile + cudagraph both work (the earlier Dynamo reshape was
  in the emulation path, which the scheme bypasses).
- **No `--speculative-config`** (see below).
- See `launch-26b-019-triton.sh` for the exact, confirmed command.

## Spec-decode (draft model) — ✅ DONE via the vLLM 0.23.0 backport
`gemma4_assistant` (Google's bespoke draft: centroids + `token_ordering` + pre/post backbone
projections) is **not** loadable via vLLM 0.19.1's Transformers fallback: it's CausalLM-only (no base
`AutoModel` class), and its composite config (`text_config` + image/audio token ids, no real vision
sub-model) is mis-classified as multimodal → vision-encoder setup fails. The fix was to **backport the
native draft-model class from vLLM 0.23.0** onto the 0.19.1 tree — `Gemma4MTPModel` /
`Gemma4AssistantForCausalLM` (2 new files) plus 5 framework patches (`eagle.py`, `speculative.py`,
`gpu_model_runner.py`, `registry.py`, `model_arch_config_convertor.py`). All staged under
[`port-0191/`](port-0191/) (`*.patch.md` + `gemma4*.py` + `launch-26b-019-spec.sh`); the 2026-06-27
3-model bench ran Gemma on this path at **MTP-3**. `--language-model-only` still clears the
*target*-multimodal gate (`_raise_if_multimodal`).

**Caveat:** MTP roughly **doubles cold prefill** (the draft cold-prefills the full context too) — for
long-context compaction use eager / no-spec or `GEMMA4_MTP_DRAFT_FULL_WINDOW=1024`; MTP's payoff is
steady-state *decode*. (The earlier `launch-26b-spec.sh` / `-ngram.sh` Transformers-fallback attempts
were dead ends — superseded by the backport.)

## Files
- `vllm-0.19.1-rdna4-nvfp4.patch` — canonical diff vs upstream 0.19.1, **3 files**: `oracle/nvfp4.py`
  (RDNA_TRITON routing/backend), `nvfp4_emulation_utils.py` (RDNA dequant + linear kernel),
  `compressed_tensors_w4a4_nvfp4.py` (RDNA linear scheme). Applies on top of the MXFP4 patch.
- `rdna_nvfp4_moe.py` — new RDNA4 NVFP4 MoE experts kernel (drop into `fused_moe/experts/`).
- `compressed_tensors_w4a4_nvfp4.py` — the ported RDNA linear scheme (also in the patch; standalone for reference).
- `rdna_nvfp4_moe.{orig,patched}.py`, `rdna_nvfp4_moe.diff` — provenance.
- `gemma4.{orig,patched}.py`, `gemma4.diff` — Gemma-4 model-class notes (`--language-model-only`).
- `BRINGUP-019.md` — 0.19.1 bring-up runbook (annotated with the as-found corrections).
- `NOTES-weightloading-gap.md` — historical weight-loading notes (0.18.1 era).
- `launch-26b-019-triton.sh` — the **confirmed** Gemma-4-26B serve wrapper.
- `routing_equiv_check.py` — CPU routing-equivalence proof.

## Next
- **Bake** (not bind-mount) for prod: fold `vllm-0.19.1-rdna4-nvfp4.patch` into `../patches` and
  `rdna_nvfp4_moe.py` into `../graft`, rebuild the image. Bind-mount overlay = the working dev rig today.
- **Upstream candidate:** RDNA NVFP4 (MoE Custom-routing + linear dequant) enables ALL NVFP4
  compressed-tensors models on RDNA — not just Gemma-4.
- **Spec-decode polish:** the `gemma4_assistant` MTP draft is backported from 0.23.0 and working at
  MTP-3 (see *Spec-decode* above); remaining work is hardening the fully-compiled (prefill + decode
  cudagraph) MTP path — decode-only cudagraph is already validated.
