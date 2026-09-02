# vLLM 0.28.0 — RDNA4 port (gfx1200/gfx1201: Radeon AI PRO R9700, RX 9070 XT)

Branch `rdna4-port-0.28.0` = upstream vLLM (0.28.0 line) + the changes needed to serve well on AMD RDNA4
consumer/workstation GPUs, which are outside the official ROCm vLLM targets (gfx90a/942/950).

Current release: **rc12** (2026-08-28) — Docker Hub **[capicua25x/vllm-rocm-rdna4](https://hub.docker.com/r/capicua25x/vllm-rocm-rdna4)**
`:0.28.0-rdna4-rc12` = `:0.28.0-rdna4` = `:latest`. Validated on 2× Radeon AI PRO R9700 (gfx1201, TP2);
gfx1200 (RX 9070 XT) reports welcome.

**Previous cycle:** the 0.26.1 line — with the full MXFP4 quantization recipe, the FP8-vs-MXFP4 two-config A/B,
short-prompt grids and the complete accuracy table — lives on
[`rdna4-port-0.26.1/RDNA4-PORT.md`](https://github.com/Capicua25x/vllm-rocm-rdna4/blob/rdna4-port-0.26.1/RDNA4-PORT.md).
Attribution and lineage are unchanged from that document.

## What's in the 0.28.0 cycle

- **Rebase onto vLLM 0.28.0**; the 0.26.1-era rc5→rc9 overlay chain is collapsed in-tree (the rc11 release
  note records the one overlay that had been left out and its recovery).
- **Hybrid GDN + full-attention models (Qwen3.5 class) hardened**: fp8 KV cache with bf16 SSM state on unified
  pages (KV pool ≈ 2× the 262k window), prefix caching in `align` mode.
- **Speculative decoding, two supported drafters**: MTP-3 (production default) and **DFlash2 k=7, including
  FP8 block-quantized drafters** — stock vLLM fails to load an fp8 drafter (`BFloat16 != Float8_e4m3fn` in the
  context-projection build); fixed in `50f7a7d12` by dequantizing the context KV projection at build time
  (per-tensor / per-channel / 128×128 block scales).
- **Tuned per-shape R9700 fp8 GEMM configs** in-tree (vLLM's own tuner, regenerated for gfx1201).
- **3D split-KV attention gate extended to speculative verify shapes**: the 2D/3D selector keys on
  `q.shape[0]` with `MAX_QLEN_3D = 8`, so an MTP/DFlash verify of small q_len over long KV takes the 3D grid
  instead of falling to a ~10-workgroup 2D launch.
- `NCCL_PROTO=Simple` recommended for TP2 over PCIe on these cards (RCCL's LL protocol measured ~2.8× slower
  for the ≈640 KB decode all-reduces).

## Serving profiles (measured, same window, 2× R9700 TP2, thinking ON)

| | **MTP-3 — production (default)** | **DFlash2-FP8 — dev-box** |
|---|---|---|
| Best for | 8+ concurrent users, mixed workloads | 1–4 users: coding assistants, interactive |
| Single-stream @6k | 55 tok/s | **65–70 tok/s (+18–26%)** |
| Under concurrency | 315 agg @c16 · ceiling ~16 users | ahead to ~c4, knee c4–8, behind at c8+ |
| Concurrent 6–7k-token requests | ~32 | ~22 (k=7 raises per-request KV reservation) |
| Draft accepted / tokens per step | 1.4–1.6 of 3 → 2.4–2.6 | 2.15 of 7 → 3.16 (c1) |
| Extra download | none (MTP head in the checkpoint) | 2.0 GB drafter |
| Runner | V1 (default) | V2 (`VLLM_USE_V2_MODEL_RUNNER=1`) |
| Output quality | same class — paired τ² parity within noise | same |

Target checkpoint for both: [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (vendor stock).
Dev-box drafter: [tcclaviger/Qwen3.8-27B-DFlash2-FP8](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8)
(community quant — credit to tcclaviger). tcclaviger also maintains his own RDNA4 vLLM stack —
[tcclaviger/vllm](https://hub.docker.com/r/tcclaviger/vllm) (closed-source, native HIP kernels, baked-in
quantizers/tuner). Complete `docker run` commands for both profiles are on the
[Docker Hub page](https://hub.docker.com/r/capicua25x/vllm-rocm-rdna4). Measured on this hardware, the V2 model
runner is equivalent to V1 in output-quality class and MTP throughput (±3% at every level, same ceiling) — the
profile difference is the drafter, not the runner.

## Serving Ornith-1.5-35B-A3B — MXFP4 MoE on gfx12

Get the quant: **[Capicua25x/Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4](https://huggingface.co/Capicua25x/Ornith-1.5-35B-A3B-MXFP4-Quark-RDNA4)** — data-free Quark MXFP4 of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (MIT), 67 GB bf16 → **21.4 GB**, with a **bundled 737 MB DFlash drafter** for single-stream serving. The model card carries the full 5-seed quality table (tracks the official FP8 within seed noise; τ²-telecom **0.965**, 110/114), the same-protocol throughput grid, and complete `docker run` commands for both profiles — the [Docker Hub page](https://hub.docker.com/r/capicua25x/vllm-rocm-rdna4) mirrors them.

The two flags that make it work on this image, and the two mistakes that silently ruin it:

- `--moe-backend triton_unfused --attention-backend TRITON_ATTN` — the quant is weight-only (`input_tensors: null`) precisely so the unfused Triton MoE lane engages. Do **not** serve MXFP4 MoE with the default (`auto`) backend on ROCm gfx12: every native MXFP4 MoE backend is CUDA/CDNA-gated and the fallback is per-call quantize-dequantize emulation (~5× slower, ~1-user ceiling).
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder` — this model family emits the qwen3-coder XML tool format; a JSON parser (hermes-style) silently turns tool calls into text and craters agentic scores (we measured τ² collapsing 0.965 → 0.14 before catching it).

Measured on 2× R9700 TP2: concurrency profile **782 agg tok/s @c32** on 6k prefill (~32-user ceiling); bundled-DFlash single-stream **92.5 tok/s** short / **64.2** @6k at c1 (not for concurrent load — the drafter's separate KV pool degrades under long-context batch pressure).

## Serving Muse-Glimmer-30B — MXFP4 dense VL on gfx12, 1M-context profile

Get the quant: **[Capicua25x/Muse-Glimmer-30B-MXFP4-Quark-RDNA4](https://huggingface.co/Capicua25x/Muse-Glimmer-30B-MXFP4-Quark-RDNA4)** — Quark MXFP4 of [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Apache-2.0): the 52 MLP blocks in MXFP4 (group 32, e8m0 scales), attention / vision tower / head in bf16, 28 GB on disk. Ships **two configs** — the default `config.json` has YaRN ×8 baked (window 1,048,576) and `config.json.bak-native131k` is the native window — plus the **bundled z-lab DFlash2 drafter** (`dflash-draft/`, Apache-2.0, unmodified).

```bash
# concurrency profile (default, 1M window — ~16 concurrent users on 2×R9700)
docker run --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface capicua25x/vllm-rocm-rdna4:0.28.0-rdna4 \
  serve Capicua25x/Muse-Glimmer-30B-MXFP4-Quark-RDNA4 --port 8011 --trust-remote-code --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 1048576 --attention-backend TRITON_ATTN --moe-backend triton_unfused \
  --enable-prefix-caching --max-num-seqs 32 --max-num-batched-tokens 8000 --max-cudagraph-capture-size 128 --skip-mm-profiling \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer \
  --default-chat-template-kwargs '{"reasoning_strength": "low"}'

# single-stream profile (bundled DFlash2 drafter, measured at the native 131k window)
#   ... --max-model-len 131072 \
#   --speculative-config '{"method":"dflash","model":"Capicua25x/Muse-Glimmer-30B-MXFP4-Quark-RDNA4/dflash-draft","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
```

- Reasoning is a chat-template kwarg, `reasoning_strength` (`low`/`medium`/`high`/`xhigh`) — `reasoning_effort` is inert on vLLM.
- The `rope_scaling` override must sit on the **text config** of this multimodal arch (a top-level override is silently ignored); the shipped config already does.
- At the 1M window, `--gpu-memory-utilization 0.92` starves the drafter's KV; 0.90 is stable on 32 GB cards.

Measured on 2× R9700 TP2 (bench v3): concurrency profile **28.4 tok/s c1 · 351 agg @c16 · 615 agg @c32** short, **328 agg @c16** on 6k prefill (~16-user ceiling at ≥20 tok/s per user); DFlash2 single-stream **57.0 tok/s c1** (accept 1.55/step) rising to **296 agg @c8** on 6k prefill (accept 2.08). Quality on the 1M config: IFEval 5-seed medians inst 0.906 / prompt 0.863 (native config 0.930 / 0.900 — the YaRN tax, stated on the card); τ²-bench telecom **0.842**, airline **0.840**. Long-context retrieval past 131k is community-verified (llama.cpp needles to 832k) — our own probe on this quant is owed and the card says so.

## Reproducibility — same config, different outputs, and why

vLLM compiles the model with inductor autotuning: candidate kernels are **benchmarked at first startup** and
the fastest is kept. That selection is a timing lottery, frozen into the compile cache
(`~/.cache/vllm/torch_compile_cache/<hash>/…`; the hash is printed at startup as `Using cache directory:`).
Measured on this port:

- Two machines pulling the same image — or one machine after a cache wipe — produce **different token streams
  at temperature 0** on long generations (~80–90% of 300-token greedy completions diverge textually). Both are
  numerically legitimate: with `--enforce-eager` (no inductor), outputs are **bit-identical** across runners
  and machines.
- Quality is unaffected within noise: across four independent draws, τ²-Bench telecom (114 tasks) spanned
  0.886–0.930 vs a 0.939 bf16 cloud reference; paired per-task differences are not statistically significant.
- **Persist the compile-cache volume** to keep a serve's outputs stable across restarts; copy it between
  machines for bit-reproducibility; treat it as part of a frozen deployment's identity.
- Debugging "same config, different outputs": compare the two serves' `Using cache directory:` hashes first.

## Accuracy / gates (0.28.0 rc12)

Cycle promotion was gated on agentic parity with the same checkpoint: τ²-telecom (114 tasks, thinking ON)
0.886–0.930 across draws vs 0.939 bf16 cloud reference, plus internal tool-calling and SQL-generation suites
at parity. The fuller table (GSM8K, IFEval, GPQA-D, AIME'25, MMLU-Pro 0.817, Terminal-Bench hard 0.341,
τ²-airline 0.86 / τ²-retail 0.82) was measured on the 0.26.1-cycle rc10 image with the same checkpoint — see
the [0.26.1 document](https://github.com/Capicua25x/vllm-rocm-rdna4/blob/rdna4-port-0.26.1/RDNA4-PORT.md).
Thinking is a chat-template kwarg (`{"chat_template_kwargs": {"enable_thinking": true}}`) — vLLM silently
ignores OpenAI `reasoning_effort`; benchmarking without thinking measurably degrades agentic scores.

## Not in this cycle (yet)

- 0.28-cycle numbers for the MXFP4 arm
  ([Capicua25x/Qwen3.8-27B-MXFP4-Quark-RDNA4](https://huggingface.co/Capicua25x/Qwen3.8-27B-MXFP4-Quark-RDNA4)
  still serves on this image; grids pending).
- gfx1200 (RX 9070 XT) validation — the cycle was validated on gfx1201 only.
- An image with the fp8-drafter loader fix baked in (rc12 predates `50f7a7d12`; the Hub page documents the
  one-file mount until the next image).

Benchmark harness: [Capicua25x/modelbench](https://github.com/Capicua25x/modelbench) (bench v4: rotating
topics, per-invocation nonce, acceptance from `/metrics` — nothing is ever regenerated).
