# vLLM on AMD RDNA4 (gfx1201) — R9700 · RX 9070 XT

> ## 🙏 Built on Rob Smith's RDNA4 work
> Everything here stands on the gfx1201 kernel enablement that **[Rob Smith](https://hub.docker.com/r/tcclaviger/vllm-rocm-mxfp4-nvfp4)** pioneered — the original MXFP4/NVFP4 + MoE kernel work for RDNA4 (his `tcclaviger/vllm-rocm-mxfp4-nvfp4` images, vLLM 0.18.x era). This repo forward-ports and extends that foundation. **Without Rob's work, none of this exists.** See `NOTICE` for the derivation chain.

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


## Models currently targeted (validated on this stack)

Hardware envelope for everything below: **2× 32 GB (64 GB total), TP2** — dual
R9700; a single 32 GB card fits the smaller quants at reduced context.

| model | quant | status |
|---|---|---|
| **Qwen3.5/3.6-35B-A3B family** (incl. hybrid GDN + MTP head) | MXFP4 (native kernel) | ✅ **in production TODAY serving [Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B)** behind a real business assistant — 111 tok/s @7k ctx w/ MTP, 262k context |
| **Muse Glimmer 30B** (multimodal) | FP8 | ✅ validated — serving, tool-calling (`muse_glimmer` parsers), agentic use; DFlash drafter integration in validation |
| **Gemma 4 26B-A4B-it** | NVFP4 (native RDNA4 MoE patches) | ✅ validated — MTP assistant backport, 256k context; full recipe in [`gemma-nvfp4/`](../../tree/rdna4/gemma-nvfp4) |
| **Qwen 3.8-27B** | FP8 planned | 🎯 targeted — image gates will extend the day weights ship |

Anything upstream vLLM runs on ROCm also works here unchanged; the value of
this fork is the native MXFP4 path and the spec-decode fixes on top.

## Serving recipes (validated configs)

Both use the published image; the fork's images have **no entrypoint** — pass it
explicitly. `--generation-config auto` (the default) adopts each checkpoint's
recommended sampling for anything the request doesn't override.

### Qwen3.5/3.6-35B-A3B family · MXFP4 · MTP spec decode (production config)

```bash
docker run --rm --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render --ipc=host \
  -v /path/to/models:/models -v hf-cache:/root/.cache/huggingface \
  --entrypoint /usr/local/bin/vllm \
  capicua25x/vllm-rocm-rdna4:glimmer-qwen38-rc5 \
  serve /models/<qwen3.5-or-3.6-35B-A3B-MXFP4> \
  --trust-remote-code --tensor-parallel-size 2 --gpu-memory-utilization 0.92 \
  --max-model-len 262144 --attention-backend TRITON_ATTN \
  --enable-prefix-caching --max-num-seqs 64 --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
```

Gotchas this config encodes (each cost a debugging session):
- **`"attention_backend":"TRITON_ATTN"` inside the speculative config is mandatory** —
  the drafter does *not* inherit `--attention-backend` (upstream design) and
  auto-selects a native paged kernel that collapses on the 1072-token pages
  hybrid models force (3.2 ms/launch).
- **`--max-num-batched-tokens 16384`**: with a speculative config present, the
  scheduler budget silently pins low and ~6k prompts split into multiple passes.
- Measured on 2×R9700: 111 tok/s at 7k-context decode, 123 short, 262k context,
  ~128-user short-prompt ceiling.

### Muse Glimmer 30B · FP8 · ATEM tool-calling

```bash
docker run --rm --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render --ipc=host \
  -v /path/to/models:/models -v hf-cache:/root/.cache/huggingface \
  --entrypoint /usr/local/bin/vllm \
  capicua25x/vllm-rocm-rdna4:glimmer-qwen38-rc5 \
  serve /models/<Muse-Glimmer-30B-FP8> \
  --trust-remote-code --tensor-parallel-size 2 --gpu-memory-utilization 0.92 \
  --max-model-len 32768 --max-num-seqs 32 --enable-prefix-caching \
  --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --default-chat-template-kwargs '{"reasoning_strength": "low"}'
```

Gotchas:
- **Both `muse_glimmer` parsers are mandatory.** Glimmer speaks the ATEM channel
  protocol (`to=self` reasoning, `<atem:invoke>` tools); with generic parsers
  (e.g. `hermes`) the channel markers leak into `content` and every structured
  task fails while *looking* fluent.
- **`reasoning_strength` defaults to `high`** in the chat template and will eat
  small `max_tokens` budgets whole (all reasoning, empty answer). `low` is the
  sane serving default; override per request via `chat_template_kwargs`.
- The official DFlash drafter (`meta-models/Muse-Glimmer-30B-assistant`, 16-token
  blocks) is supported by this branch line; our integration is still in
  validation — recipe lands here once measured.

## Want another model? Open an issue

Happy to look at adding/validating more models — [open an issue](../../issues)
with the checkpoint link. Practical constraint: it has to fit **64 GB of VRAM
total** (weights + KV cache), so realistically ≤~40B dense at FP8/W4, or MoE up
to the ~35B-A3B class at 4-bit, with context budget scaling accordingly. No
promises on timelines — this is a one-person fork-carry effort — but well-scoped
requests with a public checkpoint get tried.

## Validation summary

Full results — concurrency sweeps, the 8-bench engine A/B, and the long-context
"compaction" ladder — plus the reproduction tools live in [`validation/`](../../tree/rdna4/validation).

Engine A/B vs the previous production engine (same weights, same sampling):
greedy outputs byte-identical at 6–7k context; 7-bench quality gate (IFEval
exact tie, GPQA-D −1/60, GSM8K −1/50 & +3/50, AA-LCR +4/100, τ²-telecom
0.9825 vs 0.974, AIME parity) — no regressions; long-context survival to 255k.
Full methodology in vllm-project/vllm#51995.
