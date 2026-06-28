# Cold-prefill KV-split fix — controlled same-window A/B (2026-06-26)

Same build (the patched triton_unified_attention.py + triton_attn.py mounted), same
launch (SPEC=0, ENFORCE_EAGER=1, max_num_batched_tokens=2048, max-model-len 262144, TP2).
ONLY difference between legs: env VLLM_TRITON_3D_PREFILL (unset=baseline, =1=fix).
Cold prefill = nonce-block-0 (prefix-cache MISS, cached=0 verified). TTFT = prefill time.

| depth | base tok/s | fix tok/s |   Δ  | base TTFT | fix TTFT | time saved |
|------:|-----------:|----------:|-----:|----------:|---------:|-----------:|
| 16000 |       1804 |      1916 |  +6% |      7.4s |     6.9s |      +0.5s |
| 63000 |       1074 |      1377 | +28% |     46.6s |    36.3s |     +10.3s |
|120000 |        613 |       807 | +32% |    154.4s |   117.3s |     +37.1s |
|160000 |        472 |       614 | +30% |    266.8s |   204.9s |     +61.9s |

RESULT: KV-split prefill gives +28–32% at depth (>=63k), neutral at 16k (within noise),
NO regression anywhere. Benefit plateaus ~+30% at depth. GPU saturates (100% both cards)
under the fix at depth vs the baseline's lower-occupancy serial 2D KV sweep.

CORRECTIONS vs the earlier (single-window, uncontrolled) numbers:
- The earlier "63k -> 382 tok/s, super-quadratic collapse, ~3.6x fix" was a CROSS-WINDOW
  artifact (comparing a stale-window baseline against this window's fix). The controlled
  same-window delta is +30%, not 3.6x.
- Baseline cold prefill is NOT catastrophic: ~470–1800 tok/s, scaling ~depth^1.6
  (SUB-quadratic — diluted by the 25/30 sliding-window layers; only the 5 full layers are
  O(n^2)). The "85 tok/s / 30 min" alarm was contaminated by concurrent traffic.
- Cold-prefill throughput shows high CROSS-WINDOW variance (63k was 382 in one window,
  1074 here). Trust same-window A/B deltas, not absolute single-window numbers.

Correctness: 216/216 fp32 2D-vs-3D differential (full matrix incl. head_dim=512) + 0
failures in the full-file regression prefix + two adversarial reviews all-clear.

## Real compaction test — 157k cold one-shot (2026-06-26)

Deployable prod config: compiled FULL_DECODE_ONLY (eager prefill + graph decode) + MTP-3
+ VLLM_TRITON_3D_PREFILL=1 + VLLM_TRITON_3D_SMALL_QLEN=1. cold_prefill_test.py --target-tokens 200000.

| run        | result    | wall  | prompt_tok | canary DATO-07 | canary DATO-13 |
|------------|-----------|-------|-----------:|----------------|----------------|
| --think    | COMPLETED | 385s  |   157,159  | False*         | False*         |
| no --think | COMPLETED | 382s  |   157,156  | TRUE           | TRUE           |
* think consumed the 300-token max_tokens before answering → recall check inconclusive, NOT a read failure.

VERDICT: Gemma SURVIVES a 157k compaction — completes (no hang/deadlock) and correctly recalls
facts across the full depth. BUT ~382s (6.4 min) in the MTP-3 config. That is ~175s SLOWER than
the eager/SPEC=0 A/B (~205s @160k): MTP-3's DRAFT model also cold-prefills the full 157k context
(spec decode helps DECODE throughput, not prefill — at compaction it is pure overhead). So at
extreme depth MTP roughly DOUBLES cold-prefill time and eats most of the kernel fix's win.

Survival is therefore CLIENT-TIMEOUT-dependent: 382s passes a 400s timeout, FAILS a 200s one.
LEVER: GEMMA4_MTP_DRAFT_FULL_WINDOW caps the draft's full-attention long-context read → should cut
the draft-prefill overhead and bring the deployable config toward the ~205-250s range. NOT yet tested.
Also still open: config #1 (FULL_AND_PIECEWISE compiled-PREFILL) stall in isolation — this test used
FULL_DECODE_ONLY (eager prefill), which sidesteps that path by design.

## Draft-window lever (GEMMA4_MTP_DRAFT_FULL_WINDOW) — 157k compaction

Same deployable config + GEMMA4_MTP_DRAFT_FULL_WINDOW=1024 (caps the draft's one full-attn layer).

| config (157k cold, wall)                          | wall  | canaries | note |
|---------------------------------------------------|-------|----------|------|
| eager / SPEC=0 + fix (TTFT only, ~160k)           | ~205s | n/a      | prefill-only ref |
| compiled FDO + MTP-3 + fix                         |  382s | both ✓   | draft prefills full ctx |
| compiled FDO + MTP-3 + fix + draftwin=1024         |  327s | both ✓   | acceptance 38.5% (105/273) |

Subtracting ~20s decode (≈200 tok @ ~10 tok/s): prefill ≈ 362s / 307s vs 205s eager.
Draft-window=1024 removes ~55s of the ~157s MTP+compiled gap. The draft's single full-attn
layer was ~55s of it; the REMAINING ~100s is the draft's other layers + compiled-mode overhead,
NOT addressable by this lever. Going smaller (256) won't help much — the layer is already
157k→1024, and 1024→256 is marginal vs the 157k base.

CONCLUSION: compaction SURVIVES at ~5.5 min (draftwin) / 6.4 min (uncapped) in the MTP-3 config;
~3.4 min if MTP were off (eager). draftwin=1024 is a free -14% with no correctness/acceptance loss
→ worth adding to the prod Gemma launch. Bigger wins need dropping MTP for the prefill phase
(architectural) — fundamental tension: MTP helps steady-state DECODE but taxes cold PREFILL.

## Config #1 stall — SETTLED in isolation (open Q1 closed) 2026-06-26

Prod-candidate: compiled CUDAGRAPH_MODE=FULL_AND_PIECEWISE (compiled PREFILL) + MTP-3 + all fixes
(VLLM_TRITON_3D_PREFILL=1, VLLM_TRITON_3D_SMALL_QLEN=1, GEMMA4_MTP_DRAFT_FULL_WINDOW=1024).
157k cold one-shot, server isolated (no competing traffic), GPU sampled every 15s.

GPU utilization: 43% → 81% → 100%, then PINNED 100/100 for the entire prefill (21 samples).
Result: COMPLETED in 328s, both canaries recalled.

VERDICT: The original config-#1 "GPU-3% idle stall / looks like a deadlock" was a CONCURRENCY
ARTIFACT (concurrent load during the experiment matrix) — NOT a
real deadlock. In isolation it is GPU-100% slow-compute, identical regime to the other configs.
=> The compiled prod-candidate (FULL_AND_PIECEWISE) is VIABLE. Also proves the new 3D-prefill
data-dependent dispatch does NOT break piecewise cudagraph capture.

FULL_AND_PIECEWISE (328s) ≈ FULL_DECODE_ONLY+draftwin (327s) for cold compaction — pick between
them on DECODE throughput (separate bench), not cold-prefill cost.

=========================== FINAL VERDICT ===========================
Gemma SURVIVES context compaction. A 157k cold compaction completes + correctly recalls the full
context in ~5.5 min (compiled + MTP-3 + fixes + draftwin=1024), ~3.4 min without MTP. No deadlock
in any config tested. Recommended prod-Gemma launch flags: VLLM_TRITON_3D_PREFILL=1,
VLLM_TRITON_3D_SMALL_QLEN=1, GEMMA4_MTP_DRAFT_FULL_WINDOW=1024. Cold prefill is slow-but-usable
(rare cold-cache worst case; warm turns fast), NOT the catastrophe the contaminated alarm implied.

## Decode-throughput bench — launch-config decision (2026-06-26)

throughput_sweep.sh, 6k prefix-cached prompt (measures DECODE), max_tokens=256, levels 1/8/16.

| config                          | n=1 /user | n=8 /user (agg) | n=16 /user (agg) | spec acc |
|---------------------------------|----------:|----------------:|-----------------:|---------:|
| FDO + MTP-3 + draftwin=1024     |      92.5 |    43.1 (265)   |     33.6 (447)   |   75.9%  |
| FAP + MTP-3 + draftwin=1024     |      92.4 |    45.8 (355)   |     34.1 (441)   |   80.9%  |
| FDO no-MTP (spec off)           |      50.1 |    29.4 (234)   |     24.4 (389)   |    --    |

CONCLUSIONS:
1. MTP-3 + draftwin=1024 is a BIG decode win — +85% single-user (92.5 vs 50.1) and +38% at n=16
   (33.6 vs 24.4), at 76-81% spec acceptance (draftwin=1024 did NOT hurt acceptance). The ~2x
   cold-prefill cost (a RARE cache-miss event) is well worth the every-token decode gain. KEEP MTP-3.
2. FAP vs FDO are decode-EQUIVALENT at n=1 (92.4≈92.5) and n=16 (441≈447). FAP edges FDO at n=8
   (355 vs 265 agg) and has slightly higher acceptance (80.9 vs 75.9). Cold prefill ≈ equal
   (328≈327), config-#1 stall debunked. => Near-tie; FAP ≥ FDO everywhere measured.

RECOMMENDATION (prod-Gemma launch): VLLM_TRITON_3D_PREFILL=1 + VLLM_TRITON_3D_SMALL_QLEN=1 +
GEMMA4_MTP_DRAFT_FULL_WINDOW=1024 + MTP-3, with CUDAGRAPH_MODE=FULL_AND_PIECEWISE (slight edge,
no downside now stall is debunked). Conservative fallback = FULL_DECODE_ONLY (eager prefill =
less compiled surface) — decode-equivalent. The n=8 FAP>FDO gap (265 vs 355) is the only
non-parity point and could be measurement variance; a re-run would confirm if decision-critical.

## n=8 gap CONFIRMED (reversed order + 3 repeats) 2026-06-26

Re-ran FAP first / FDO second (reversed vs original) ×3 each, same server (no reload between reps).
KEY: the FIRST repeat of each config is COLD (low) — must discard. Warm (rep 2,3) aggregate tok/s:

| level | FAP warm | FDO warm | FAP edge |
|------:|---------:|---------:|---------:|
|   4   |   ~337   |   ~276   |   +22%   |
|   8   |   ~388   |   ~344   |   +13%   |
|  12   |   ~489   |   ~430   |   +14%   |

FAP beats FDO by ~13-22% at moderate concurrency, even though FAP ran FIRST (thermal disadvantage)
=> REAL config effect, not order/thermal. The original 265-vs-355 (34%) gap was inflated by FDO's
cold-first-run (265); true edge ~13%. CONFIRMS FAP as the prod-Gemma compiled mode.
Methodology note: always warm up / discard the first bench repeat (cold-start skews it low).

FINAL launch config (locked): CUDAGRAPH_MODE=FULL_AND_PIECEWISE + MTP-3 + GEMMA4_MTP_DRAFT_FULL_WINDOW=1024
+ VLLM_TRITON_3D_PREFILL=1 + VLLM_TRITON_3D_SMALL_QLEN=1.

## ⚠️ 256k HEAD-TO-HEAD: Gemma vs Qwen — Gemma FAILS (2026-06-26)

> **⛔ SUPERSEDED 2026-06-27 — see [`BLOCKER2-fp8-capacity-RESOLVED.md`](BLOCKER2-fp8-capacity-RESOLVED.md).**
> The "admission deadlock" below was NOT a fundamental Gemma/scheduler limit — it was the **fp16
> KV-cache capacity ceiling** (vLLM hung waiting for KV blocks instead of cleanly rejecting when the
> cache was exhausted). With `--kv-cache-dtype fp8` (333k-token cache > 262k max-model-len), Gemma
> **admits + cold-prefills ≥256k, no deadlock, recall 20/20**. The analysis below is kept as the
> investigation trail; its "Gemma does not match Qwen / ceiling ~175k" conclusions are obsolete.

Put Gemma on :8011 as the served model, fill ~256k, compare to Qwen. Multi-needle recall (20 canaries, needle256k.py).

| run                                  | depth   | result      | GPU      | recall | notes |
|--------------------------------------|---------|-------------|----------|--------|-------|
| QWEN distill (prod)                  | 258k    | ✅ 158s     | 100%     | 20/20* | reference — clean |
| Gemma FAP + MTP3 + draftwin + 3Dfix  | 247k    | ❌ STALL     | 3%       | —      | 11min no progress |
| Gemma FDO + MTP3 + draftwin + 3Dfix  | 247k    | ❌ STALL     | 4%       | —      | eager prefill — rules out compiled-prefill |
| Gemma FDO + MTP3 + draftwin, 3Dfix OFF| 247k   | ❌ STALL     | 3%       | —      | NOT my 3D fix |
| Gemma FDO, SPEC=0 (no MTP), no 3Dfix | 247k    | ❌ STALL     | 3%       | —      | NOT MTP either |
| Gemma FDO no-MTP                     | ~170k   | ✅ computing | 100%     | —      | works (slow); cut at 400s timeout |
* Qwen 17/20 by the strict checker; the 3 "misses" (DATO-09/15/19) were correct bare values (7, 48, 15).

ROOT CAUSE: Gemma's engine cannot ADMIT a ~247k request — prompt_tokens_total stays 0.0,
num_requests_running stays 0, GPU 3%, workers ~195% CPU. The prefill NEVER STARTS. This is an
ADMISSION/SCHEDULING DEADLOCK, not compute and NOT capacity (KV cache 160k tokens = "6.58x
concurrency for 262144 tokens/req" → a 262k request only costs ~24k slots; 25/30 layers are
sliding-window). It is INDEPENDENT of MTP and of the VLLM_TRITON_3D_PREFILL fix (stalls with both off).
The cliff is between ~170k (GPU 100%, computes) and 247k (GPU 3%, deadlock). This IS the original
"config #1 stall / engine not dispatching" — context-length-gated, reproduces in isolation.

IMPLICATIONS / CORRECTIONS to earlier today:
- "Gemma survives a compact" / "compaction resolved" / "FAP recommended" were all validated only at
  <=160k. At >~170k Gemma is much SLOWER than Qwen (Qwen 258k=158s; Gemma ~170k needs >400s) AND
  hits a hard admission deadlock before 256k. Qwen handles 256k cleanly; GEMMA DOES NOT MATCH QWEN.
- For long-context use of :8011, Qwen remains the better model. Gemma's give-back recipe is sound at
  moderate context but has an unresolved large-context (>~170-247k) admission deadlock on vLLM 0.19.1/RDNA4.
- OPEN: exact deadlock cliff (binary search 170k<->247k); root-cause the scheduler admission spin
  (likely vLLM chunked-prefill / sliding-window block-manager bug at large context).

## CLIFF FOUND — Gemma usable-context ceiling ≈ 170-180k (2026-06-26)  ⛔ SUPERSEDED 2026-06-27

> The ~175k ceiling was the **fp16 KV-cache capacity**, not a hard Gemma limit. Fixed by
> `--kv-cache-dtype fp8` (333k cache > 262k max-model-len) — Gemma cold-prefills 256k. See
> `BLOCKER2-fp8-capacity-RESOLVED.md`. The binary-search analysis below is retained as the trail.

Binary search (probe_cliff.py, admit = num_requests_running>=1, same server, ~80s/probe):
  ~169k actual: ADMIT (GPU 100%) | ~180k: DEADLOCK | ~191k: DEADLOCK | ~214k: DEADLOCK
=> ADMITS up to ~169k, DEADLOCKS at ~180k. Hard ceiling ≈ 175k tokens.

The cliff (~175k) ≈ the 160k KV-cache size → the ceiling IS capacity-bound by the 5 FULL-attention
layers (head_dim 512, store full context). The "Maximum concurrency 6.58x for 262144" startup line is
MISLEADING — it averages in the 25 cheap sliding-window layers; the full-attention layers cap real
context at ~the KV-cache size. Two distinct issues:
  1. CAPACITY: Gemma max context ≈ 175k @ gpu-mem 0.92/TP2 (vs Qwen 316k cache → 256k+). Gemma's
     head_dim-512 full layers are KV-expensive. Would need more VRAM or fewer/cheaper full layers.
  2. vLLM BUG: a request beyond the ceiling DEADLOCKS (scheduler spins, workers 195% CPU, prompt_tokens
     stays 0) instead of cleanly rejecting with "context too long". Worth an upstream report.

PRACTICAL TAKEAWAY: Gemma is FINE for the real workloads (chatbot ~6-40k, short-prompt chat/analysis
— all << 170k). It only fails as a NEAR-MAX-CONTEXT (>175k) model. For long-context, Qwen (256k) wins.
UNCONFIRMED: whether raising gpu-mem-util moves the cliff up (would prove capacity-bound) — 1 test if wanted.
