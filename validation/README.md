# Validation results — rc5 engine on 2× Radeon AI PRO R9700 (gfx1201, TP2)

**This stack is not a demo: it serves
[Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B)
(35B-A3B hybrid MoE, MXFP4, MTP spec decode) in production today**, behind a
business assistant handling real daily traffic. Everything below was measured
on the production hardware. Tools in this directory reproduce the methodology.

## Concurrency sweeps (256-token answers, temp 0)

~7k-token cacheable prefix ("long system prompt" shape):

| users | per-user tok/s | aggregate |
|---|---|---|
| 1 | 111.7 | 112 |
| 8 | 45.5 | 356 |
| 16 | 29.1 | 454 |
| 24 | 23.6 | 555 |
| 32 | 18.9 | 590 |
| 64 | 11.0 | 605 |

Short prompt (compute-bound ceiling):

| users | per-user tok/s | aggregate |
|---|---|---|
| 1 | 123.3 | 123 |
| 16 | 61.4 | 897 |
| 32 | 42.1 | 1261 |
| 64 | 30.8 | 1427 |
| 128 | 22.0 | 1551 |

Practical ceiling at a 20 tok/s per-user floor: **~128 users** short-prompt,
**~24** at the 7k shape.

## Engine A/B quality gate (this fork's engine vs the previous production engine)

Same weights, same sampling, anchor conditions replicated; all denominators
audited (n as requested, empties counted, sims closed).

| bench | n | previous engine | this engine |
|---|---|---|---|
| IFEval | 80 | 0.925 | **0.925** (exact) |
| GPQA-Diamond | 60 | 0.7833 | 0.7667 (−1 item) |
| GSM8K (no-think) | 50 | 0.94 | 0.92 (−1 item) |
| GSM8K (thinking) | 50 | 0.60 | **0.66** (+3) |
| AA-LCR long-context | 100 | 0.63 | **0.67** (+4, same LLM judge both sides) |
| τ²-bench telecom (agentic) | 114 sims | 0.974 | **0.9825** (112/114, all sims closed) |
| AIME'25 | 30 | 0.000 (0/30 terminate) | 0.000 (12/30 terminate) |
| HLE (first cell, no anchor) | 120 | — | 0.2167 (19 non-terminating items score 0) |

Greedy outputs are **byte-identical** between engines at 6–7k context; MTP
draft acceptance unchanged (2.9–3.3). A 166-test domain regression suite
(SQL/BI assistant workload) passes with 0 failures.

## Long-context survival ("compaction test")

`context-ladder.py` — single prompts approaching the context limit
(max_model_len 262144):

| prompt tokens | latency | prefill tok/s | result |
|---|---|---|---|
| 40,441 | 8.2 s | ~5,700 | coherent summary |
| 127,434 | 41.9 s | ~3,100 | coherent summary |
| **254,851** | 138.7 s | ~1,850 | **coherent summary at 97% of the window** |
| over limit | — | — | clean HTTP 400, no crash |

Adversarial chunked-prefill edge (final scheduler chunk ≤8 tokens, the shape
the small-q split-KV fix newly routes): runs cold and replays warm
**deterministically** (`determinism-probe.py`: cold == warm, temp 0).

Known residue, disclosed: cold prefill is ~0.69× the previous engine
(7.1k vs 10.4k tok/s single-stream); under continuous batching long prefills
reach ~12.7k tok/s. High-concurrency (≥24 users) at the 7k shape trails the
previous engine by ~8–13%.
