# vLLM 0.26.1 — RDNA4 port (gfx1200/gfx1201: Radeon AI PRO R9700, RX 9070 XT)

Branch `rdna4-port-0.26.1` = upstream vLLM 0.26.1 + the changes needed to serve well on AMD
RDNA4 consumer/workstation GPUs, which are outside the official ROCm vLLM targets (gfx90a/942/950).

## Attribution and lineage

The gfx1201 enablement this port descends from was first done by **Rob Smith (`tcclaviger`)** for the
vLLM **0.18.1** line and shipped as `tcclaviger/vllm-rocm-mxfp4-nvfp4` — the working RDNA4 base
(gfx1201 hipBLASLt + MXFP4/NVFP4 MoE kernels). Chain: his 0.18.1 work → forward-port to **0.19.1** in
`Capicua25x/vllm-rocm-rdna4-legacy` (archived) → this **0.26.1** branch. Separately, `rdna_fp8.py`
takes its design lineage from his `_matmul_fp8_ogs` (0.24 line, W8A8): the same per-K-group scale fold
on WMMA v2.

His source is **no longer publicly available** (as of 2026-08-16 his RDNA4 work ships as the
`tcclaviger/vllm` image, weights on Hugging Face). Apache-2.0 does not require source distribution and
the grants under which the earlier source was received are unaffected — this note exists so the record
survives, not as a complaint.

What is **this project's** work, for the avoidance of doubt: the MXFP4 nibble→e4m3 unpack and the
three-regime dispatch in `rdna_fp8.py`; the spec-decode verify fix (`MAX_QLEN_3D`); the ROCm fp8-KV
attention overlay (fp8 query input, so K/V are not dequantized inside the KV loop); the clean-room
head-dim-512 flash-prefill kernel (blueprint from llama.cpp, MIT, no code copied); model bring-up,
serving recipes and the validation campaigns. See `NOTICE`.

## Quantization policy (2026-08-16)

**4-bit: AMD Quark / MXFP4 only.** Everything quantized in-house goes through Quark and ships as MXFP4; the RTN
pipeline is retired for new work. **NVFP4 is not a target on this port** — an RDNA4 NVFP4 implementation exists for
vLLM 0.19.1 in `Capicua25x/vllm-rocm-rdna4-legacy` (archived, Gemma-4 MoE + head-512 flash prefill) and is
deliberately not forward-ported; revisit only if a required checkpoint is NVFP4-only. On RDNA4 there is no FP4
datapath, so both formats are "unpack E2M1 into something the WMMA unit eats" — the difference is scale handling
(MXFP4: E8M0 per 32; NVFP4: e4m3 per 16 + tensor fp32), i.e. a scale change in one kernel, not a new port.

**8-bit: the vendor's own FP8** (e.g. `Qwen/Qwen3.8-27B-FP8`) — native FP8 WMMA on gfx1201, no in-house work needed.

**Consumption is unrestricted**: the port loads what the ecosystem publishes (Quark, compressed-tensors, AWQ/GPTQ,
vendor FP8). The policy above is about what this project *produces*.

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
  Effect on Qwen3.8-27B MXFP4 TP2 (2× R9700), **all think-OFF (raw completions)**: single-stream 51 → **61 tok/s**; short sweep c1 57 (= stock
  FP8), c32 aggregate 649 (old 600, FP8 430); 6k-prefill c8 29 (old 22, FP8 32). **Think-ON is a different, slower
  shape — same box, 2026-08-16: c1 46.5, c16 384, c32 531; do not compare think-OFF and think-ON numbers.**
  gsm8k n=50 ×3 seeds and
  the 166-test analista suite unchanged vs the old kernel. Design lineage: Rob's `_matmul_fp8_ogs`
  (0.24 line, W8A8) — same per-K-group scale fold on WMMA v2; this one adds the MXFP4 unpack.
- Triton unified-attention: allow the 3D split-KV path for small-q spec-decode verify
  (`MAX_QLEN_3D=8`) — restores MTP/DFlash/DSpark verify throughput on gfx1201
  (branch `fix/unified-attn-3d-smallq` carries this alone).
- ROCm/gfx1201 build + container recipe (image below), FP8 via native FP8 WMMA.

## Image (Docker Hub)
`capicua25x/vllm-rocm-rdna4:0.26.1-rdna4-rc6` (digest `sha256:50701299…`; = `:0.26.1-rdna4` = `:latest`
after the 2026-08-15 gate) — rc6 = rc5 + `RdnaMxfp4Fp8LinearKernel`. rc5 = `sha256:0f5cbc40…`
(also tagged `glimmer-qwen38-rc5`, kept).
Previous generation: `:0.19.1`.

## Models validated on this port (2× R9700, TP2 unless noted)
| Model | Format | Spec-decode | Notes |
|---|---|---|---|
| Qwen3.8-27B (dense hybrid GDN/attn, VL, native MTP) | FP8 (stock) / **MXFP4** (ours) | MTP-3 | MXFP4: 262k window, ~61 tok/s think-OFF (rc6; 51 on rc5) / ~46 think-ON; FP8: 64k, ~63 tok/s think-OFF. Recipe: github.com/Capicua25x/qwen3.6-mxfp4-rdna4 |
| Qwen3.6-35B-A3B distills (Ornith-1.0-35B, DSV4Pro-Thinking) | MXFP4 (compressed-tensors) | MTP-3 (grafted head) | ~75–107 tok/s single-stream; production engine 2026-06 → 2026-08-15 |
| Muse-Glimmer-30B | MXFP4 / FP8-block | DFlash draft (z-lab) | dense; the first model brought up on this line (hence the old tag name) |
| RadixArk Qwen3.8-27B-DSpark | bf16 draft | DSpark block-7 (V2 model runner) | works; loses to native MTP-3 on this hardware (44 vs 63 tok/s) |
| amd/Qwen3.8-27B-Quark-AWQ-MXFP4 | Quark W4A4 | MTP-3 (with `mtp.*` exclude patch) | runs as W4A16 on the RDNA kernel |

## Not here
No PRs upstream by choice — the branch is carried on this fork (patches are separable:
`feat/rdna4-mxfp4-linear`, `fix/unified-attn-3d-smallq`).
