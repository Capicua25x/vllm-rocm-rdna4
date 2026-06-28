# Multi-model bench — Qwen base vs Qwen distill vs Gemma-4 on RDNA4 (vLLM 0.19.1, 2× R9700 TP2)

Full capability + throughput battery across three models on the **same** stack, each served under the
alias `qwen` on `:8011` so one harness compares them apples-to-apples. Run 2026-06-27.

**Hardware / runtime:** 2× AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB ea.), tensor-parallel 2,
ROCm, vLLM 0.19.1, `TRITON_ATTN` backend. Common serve flags: `--max-model-len 262144`,
`--gpu-memory-utilization 0.92`, `--max-num-seqs 64`, `--enable-prefix-caching`. All three run **MTP-3
spec-decode** (output-lossless) and are **compiled** (decode), not eager.

## Models

| tag | model | KV | spec | notes |
|---|---|---|---|---|
| **base** | `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` | native | MTP-3 | stock Qwen MoE |
| **distill** | `Capicua25x/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4-Vision` | native | MTP-3 | DeepSeek-V4-Pro thinking distill, vision, `trust-remote-code` |
| **gemma** | `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | **fp8** | MTP-3 (`gemma4_assistant` draft) | full-cap 256k config: head-512 flash kernel (`VLLM_FA_HEADCHUNK=64`) + fp8 KV (measured 304,032 tokens; > 262k max-model-len) + `CUDAGRAPH_MODE=FULL_DECODE_ONLY` (compiled decode, eager prefill) |

The Gemma config is the locked recipe from the two RDNA4 blockers: **#1** the head-512 flash-prefill kernel
(`gemma-nvfp4: head-512 flash-prefill kernel`, ~3.5× cold prefill) and **#2** fp8 KV lifting cache capacity
above `max-model-len` so 256k admits without the prior ~175k deadlock (see
`../cold-prefill-debug/BLOCKER2-fp8-capacity-RESOLVED.md`).

## Results

### Single-stream decode (short prompt, n=1)

| model | tok/s |
|---|---:|
| base | 100.1 |
| distill | 108.8 |
| **gemma** | **109.0** |

All three land ~100–109 tok/s single-user with MTP-3. **Gemma matches Qwen on the headline single-stream number.**

### Concurrency — short prompt (decode-bound)

| n | base per-user / agg | distill | gemma |
|---:|---:|---:|---:|
| 1 | 100.1 / 100 | 108.8 / 109 | 109.0 / 109 |
| 16 | 48.7 / **739** | 49.3 / **761** | 19.4 / **226** |

Qwen scales far better on short/decode-bound batches (≈760 agg @16 vs Gemma's 226). Gemma's heavier
per-token compute (head-512 full-attention layers + NVFP4 MoE) plus spec-decode overhead that doesn't
pay off once the batch already saturates the GPU costs it here.

### Concurrency — 6k prompt (prefill-heavy)

| n | base per-user / agg | distill | gemma |
|---:|---:|---:|---:|
| 1  | 82.8 / 83  | 81.1 / 81  | 38.2 / 38 |
| 16 | 29.9 / 459 | 29.4 / 457 | 28.8 / 332 |
| 32 | 19.6 / 591 | 19.0 / 579 | 23.7 / **635** |
| 64 | 12.0 / 704 | 11.6 / 687 | 15.8 / **816** |

The picture **inverts** at high concurrency: Gemma's NVFP4 MoE prefill batches efficiently, so at n=32/64
its aggregate (635 / 816) **exceeds** both Qwen variants (≤591 / ≤704). Gemma's weak spot is single-user 6k
(38 vs ~82) — its 6k prefill is slow per request, but that cost amortizes across a full batch.

### Cold max-context prefill (unique nonce = cache miss, `max_tokens=8` → wall ≈ raw prefill)

| model | prompt tokens | wall | prefill tok/s |
|---|---:|---:|---:|
| base | 212,527 | 103.9s | ~2045 |
| distill | 210,139 | 101.6s | ~2068 |
| **gemma** | 205,300 | **320.0s** | ~642 |

Gemma cold-prefills ~3.1× slower than Qwen — the structural cost of its head-512 full-attention layers
(Qwen's per-token KV is tiny). **But with the head-512 flash kernel it does ~205k in 320s — comfortably
under the 600s compaction bar.** That is the entire point of blocker #1: without the kernel this
number is far worse and 256k compaction would time out.

### Compaction round-trip (long ctx → summarize 400 tok → 1 continue turn)

| model | ctx tokens | summarize | continue | **total** |
|---|---:|---:|---:|---:|
| base | 153,724 | 67.8s | 1.5s | **69.4s** |
| distill | 152,041 | 64.6s | 0.9s | **65.5s** |
| **gemma** | 146,313 | 217.8s | 2.5s | **220.3s** |

Same prefill-dominated story (Gemma ~3.2× Qwen) — **all three well under 600s**. The continue turn is fast
everywhere (compiled decode + MTP).

### Vision

| model | supported | wall |
|---|:---:|---:|
| base | ✅ | 3.4s |
| distill | ✅ | 3.3s |
| **gemma** | ✅ | **0.5s** |

All three describe the test image correctly. Gemma is fastest via its native vision path.

### Audio — UNTESTED

Not exercised in this battery. The Qwen variants are text+vision; Gemma-4's audio path (if the omni
model exposes one on this stack) is **not wired or validated in this recipe**. The bench leg was
skipped → **audio is untested**, not claimed either way.

### Downstream task-accuracy regression (137 cases)

A downstream task-accuracy regression suite (137 cases, NDA, withheld) driven through each model
on `:8011`. (Raw suite content is withheld under NDA and is not committed; only the scorecard is
reported — see each `results/<tag>/regression_summary.txt`.)

| model | PASS | WARN | FAIL |
|---|---:|---:|---:|
| base | 134 | 3 | **0** |
| distill | **135** | 2 | **0** |
| gemma | 133 | 4 | **0** |

**All three: zero hard failures.** Every WARN is a soft hint-substring miss on a semantically-correct
answer. The distill leads (it's the tuned model); base is within 1; **Gemma lands within 2 of the
tuned distill as a drop-in — on prompts whose few-shots, reasoning tags, and `enable_thinking` flag are all
Qwen-shaped.** That cross-family robustness is the key result — it shows the RDNA4 enablement isn't
tied to one model family.

## Verdict — a second model family is viable on RDNA4 (not just Qwen)

Gemma-4-26B-A4B-NVFP4 is a genuine second model family on this stack — a drop-in alongside Qwen:

- **matches** Qwen on single-user decode (~109 tok/s) and on downstream task accuracy (0 FAIL, within 2 of the tuned distill),
- **exceeds** Qwen on high-concurrency prefill-heavy load (n=32/64),
- **fastest** vision,
- its one structural weakness — ~3× slower cold prefill from head-512 full-attention — is **mitigated by the
  head-512 flash kernel** to stay under the 256k compaction bar (320s ≪ 600s).

The trade it asks for: weaker short-prompt, decode-bound concurrency (≈226 vs ≈760 agg @16). So the routing
guidance is profile-dependent — **single-user or prefill-heavy multi-tenant → Gemma is fully competitive;
high-concurrency short-prompt chat → Qwen stays stronger.**

## Caveats

- **KV dtype differs by model** (each at its best-fit): Gemma fp8 — **this bench run (compiled decode +
  MTP-3) measured a GPU KV cache of 304,032 tokens** at startup (17.4 GiB; 12.45× max concurrency at
  262k). The **eager / no-spec ceiling is higher — a measured 333,056 tokens** (`BLOCKER2`, what admits a
  cold 256k); the ~29k delta is the cudagraph capture + the draft model's VRAM. Both sit well above the
  262k max-model-len. Qwen native (Qwen's tiny per-token KV already yields ~390k cache, so fp8 buys it
  nothing). Not an apples-identical KV setting — it's each model's intended config.
- **6k n=64**: Qwen's ~390k cache fits 64×6k=384k; Gemma's 304k is exceeded by 384k → mild preemption.
  Gemma still leads aggregate there, so this doesn't change the ranking.
- **One run per cell.** fill/compact are cold single-shot; the downstream suite is non-deterministic (`top_p=0.95`) so
  ±1–2 on the scorecard is noise. Regression wall-clock differs (Gemma ~20 min vs Qwen ~9 min) purely from
  slower per-call decode/prefill.

## Reproduce

```bash
# Qwen base / distill (served as 'qwen' on :8011):
VARIANT=base    ./launch-bench-qwen.sh     # or VARIANT=distill
# Gemma full-cap 256k config (served as 'qwen' + 'gemma'):
./launch-bench-gemma.sh
# then, once "Application startup complete":
./bench_battery.sh <tag>                    # image,audio,fill,compact,regular,6k
#   set REGRESSION_CMD=... to add a downstream regression leg (off by default)
```

Harness:
- `launch-bench-qwen.sh` / `launch-bench-gemma.sh` — serve one model under alias `qwen` on `:8011`.
- `bench_battery.sh <tag>` — runs all legs, writes `results/<tag>/`.
- `fill_compact_image.py` — the timed fill / compact / image / audio legs (unique nonce per long-ctx leg → cold, no prefix-cache contamination).
- `throughput_sweep.sh` — per-user + aggregate tok/s concurrency sweep.
