# vLLM on AMD RDNA4 (gfx1201) — R9700 · RX 9070 XT

This fork carries production-validated improvements for running vLLM on
**RDNA4** GPUs (Radeon AI PRO R9700, Radeon RX 9070 XT — `gfx1201`), maintained
as rebased branches rather than upstream PRs (fork-carry model): use them, fold
them upstream, or supersede them freely.

## Branches

| branch | what | status |
|---|---|---|
| [`fix/unified-attn-3d-smallq`](../../tree/fix/unified-attn-3d-smallq) | Allow the 3D split-KV attention path for speculative-decode **verify** shapes (q_len 2–8). Recovers an **89% decode-throughput regression** at long context with MTP/EAGLE-style spec decoding. ~10 lines, wrapper-only. | validated in production; was vllm-project/vllm#51995 (closed by author — history & full validation there) |
| [`feat/rdna4-mxfp4-linear`](../../tree/feat/rdna4-mxfp4-linear) | Native **MXFP4 dense linear kernel** for gfx1201 via `triton_kernels` `matmul_ogs` (plus MoE routing), replacing full-weight emulation. Self-gates on an RDNA-enabled triton_kernels toolkit; falls back to emulation otherwise. | numerically validated (bit-exact dequant; GEMM Frobenius ≤4e-10) |

## Ready-to-run image

`capicua25x/vllm-rocm-rdna4:glimmer-qwen38-rc5`
(`sha256:0f5cbc404ae944783ae55fa48811f6653ceee346a7994ca5481fcba213b2f0df`) —
both branches baked, serving a 35B-A3B MXFP4 hybrid at **111 tok/s** (7k-context
decode, MTP k=3) / **123 tok/s** short, TP2 across two R9700s.

Note: spec-decode configs must pin the drafter's backend explicitly —
`--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'` —
the drafter does not inherit `--attention-backend` (upstream design).

## Validation summary

Engine A/B vs the previous production engine (same weights, same sampling):
greedy outputs byte-identical at 6–7k context; 7-bench quality gate (IFEval
exact tie, GPQA-D −1/60, GSM8K −1/50 & +3/50, AA-LCR +4/100, τ²-telecom
0.9825 vs 0.974, AIME parity) — no regressions; long-context survival to 255k.
Full methodology in vllm-project/vllm#51995.
