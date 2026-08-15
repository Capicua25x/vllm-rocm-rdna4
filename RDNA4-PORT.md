# vLLM 0.26.1 — RDNA4 port (gfx1200/gfx1201: Radeon AI PRO R9700, RX 9070 XT)

Branch `rdna4-port-0.26.1` = upstream vLLM 0.26.1 + the changes needed to serve well on AMD
RDNA4 consumer/workstation GPUs, which are outside the official ROCm vLLM targets (gfx90a/942/950).

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
  Effect on Qwen3.8-27B MXFP4 TP2 (2× R9700): single-stream 51 → **61 tok/s**; short sweep c1 57 (= stock
  FP8), c32 aggregate 649 (old 600, FP8 430); 6k-prefill c8 29 (old 22, FP8 32). gsm8k n=50 ×3 seeds and
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
| Qwen3.8-27B (dense hybrid GDN/attn, VL, native MTP) | FP8 (stock) / **MXFP4** (ours) | MTP-3 | MXFP4: 262k window, ~61 tok/s (rc6; 51 on rc5); FP8: 64k, ~63 tok/s. Recipe: github.com/Capicua25x/qwen3.6-mxfp4-rdna4 |
| Qwen3.6-35B-A3B distills (Ornith-1.0-35B, DSV4Pro-Thinking) | MXFP4 (compressed-tensors) | MTP-3 (grafted head) | ~75–107 tok/s single-stream; production engine 2026-06 → 2026-08-15 |
| Muse-Glimmer-30B | MXFP4 / FP8-block | DFlash draft (z-lab) | dense; the first model brought up on this line (hence the old tag name) |
| RadixArk Qwen3.8-27B-DSpark | bf16 draft | DSpark block-7 (V2 model runner) | works; loses to native MTP-3 on this hardware (44 vs 63 tok/s) |
| amd/Qwen3.8-27B-Quark-AWQ-MXFP4 | Quark W4A4 | MTP-3 (with `mtp.*` exclude patch) | runs as W4A16 on the RDNA kernel |

## Not here
No PRs upstream by choice — the branch is carried on this fork (patches are separable:
`feat/rdna4-mxfp4-linear`, `fix/unified-attn-3d-smallq`).
