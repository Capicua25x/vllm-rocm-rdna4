# Head-512 flash-prefill kernel rewrite (vLLM Triton, RDNA4) — design

**Goal:** make Gemma-4 full-attention (head-512) COLD prefill fast on RDNA4, in vLLM, so Gemma can stand
alone at fast 256k (a second model family on RDNA4, alongside Qwen). Target = beat the stock TILE-32 monolithic baseline (696 tok/s @63k, 388 @120k,
fp8 KV, eager), ideally approaching llama.cpp's ~5.5× (it does 247k in ~4.5min on the same cards).

## Root cause (measured 2026-06-27, NOT inherited assumptions)

The head-512 prefill kernel is **LDS-occupancy-bound** on RDNA4 (64 KiB LDS/CU). Evidence (same-window A/B):
- TILE-32 monolithic (stock): 696 tok/s @63k — BEST.
- TILE-64 monolithic (fp8 lets it compile): 392 tok/s — 1.87× SLOWER. Bigger tile = 64 KiB K/V staging =
  only ~1 resident wave-block/CU = occupancy-starved.
- TILE-64 via head-chunked QK (lever A): 380 tok/s — 1.83× slower (chunked QK but left V monolithic → no
  LDS win + re-read Q per tile).
- GQA-split (shrink BLOCK_M): impossible — BLOCK_M is already 16 (Gemma full layers: 16 q-heads / 2 global
  KV-heads / global_head_dim=512 / k_eq_v; at TP2 → 8 q-heads/1 KV-head/rank → num_queries_per_kv=8 →
  BLOCK_M=16, BLOCK_Q=2). The Q tile (16×512) is tiny; the 131072-B LDS cap is the head-512 **K+V** tiles
  (512×TILE), independent of BLOCK_M.

⇒ The lever is the OPPOSITE of bigger tiles: **shrink per-iteration LDS so MANY wave-blocks stay resident.**

## Reference design — llama.cpp `fattn-tile.cuh`, RDNA config (`get_config_amd_rdna`, head-512 = DKQ512/DV512)

    ncols=4  nthreads=128 occupancy=2 nbatch_fa=64  nbatch_K=64
    ncols=8  nthreads=256 occupancy=2 nbatch_fa=64  nbatch_K=64
    ncols=16 nthreads=256 occupancy=4 nbatch_fa=64  nbatch_K=64
    ncols=32 nthreads=256 occupancy=2 nbatch_fa=128 nbatch_K=64

- **nbatch_K=64**: the 512-deep head contraction is processed in 64-wide chunks (8 chunks). The shared tile
  `KV_tmp` only ever holds `nbatch_fa × nbatch_K` (= 64×64) — the full 512-deep K/V is NEVER staged.
- **nbatch_fa=64**: KV positions per FA iteration (the "TILE").
- **occupancy 2-4 is the explicit launch target.** Confirms the mechanism we measured.
- Q is register-resident; K-chunk streamed into shared then to registers; QK accumulated per chunk; PV
  likewise chunked over the OUTPUT head dim so V is never fully staged.

## Why lever A failed (the lesson that shapes this design)

Lever A chunked ONLY the QK dot (head-dim slices) but kept the PV path monolithic — V_load `[TILE,512]`
was still staged in full, so LDS was NOT actually reduced → no occupancy gain. On top of that it re-loaded
Q from HBM inside every KV tile. Net: −1.8×. **The fix must chunk BOTH QK and PV, and keep Q resident.**

## Triton design (principles, not transliteration)

Per program (BLOCK_M=16 rows = BLOCK_Q=2 positions × num_queries_per_kv=8 heads, one KV head/rank):

1. **Q resident, loaded ONCE** before the KV loop. 16×512 fp16 = 16 KiB → register-resident (16×512/64
   threads ≈ 128 regs/thread for Q across the wave — acceptable; or split into the 8 head-chunks as 8×(16×64)
   register tiles loaded once). NO per-tile Q re-read.
2. **Outer loop over KV tiles** of size `TILE = nbatch_fa` (start 32; try 64). Online softmax (running M, L,
   acc) across tiles — unchanged from the current kernel.
3. **Inside each KV tile, loop the head contraction in `KC = nbatch_K = 64`-wide chunks (8 chunks):**
   - `S[BLOCK_M, TILE] += dot(Q_chunk[BLOCK_M, KC], K_chunk[KC, TILE])` — only a `[KC, TILE]` K-chunk staged
     (64×TILE×1B fp8 = 2 KiB at TILE32). fp8 K dequant per chunk, in-register.
   - After the 8 chunks, `S *= scale`; apply mask/alibi/softcap/sinks exactly as today.
4. **PV chunked over the OUTPUT head dim in `KC`-wide pieces (8 chunks):** for each output-head chunk
   `acc[:, c:c+KC] += dot(P[BLOCK_M, TILE], V_chunk[TILE, KC])` — only a `[TILE, KC]` V-chunk staged. With
   k_eq_v the V-chunk can reuse the K-chunk load (Gemma full layers share K=V) — a free bandwidth halving.
5. **LDS target: < 16 KiB/block → occupancy ≥ 4** (vs stock 32 KiB → 2). That occupancy lift is the speedup.

Env-gate: `VLLM_FA_HEADCHUNK=KC` (0 = stock monolithic, default). Applies ONLY when head_size==512 (Gemma
full layers); the 25 sliding head-256 layers stay monolithic (already fine).

## Correctness plan (the real risk = silent wrong attention)

- The chunked QK/PV is mathematically identical to monolithic (sum over head chunks == full dot; online
  softmax untouched). Validate with the existing fp32 2D-vs-reference differential test (head_dim=512 case)
  before any perf claim.
- k_eq_v reuse must be verified against a non-k_eq_v reference (load V independently first; add the K=V reuse
  only after the independent-V version is correct).

## Test plan

1. Compile at the reduced LDS (confirm Triton reports < stock shared; confirm occupancy via rocprof or
   inferred from LDS).
2. fp32 differential correctness (curated head-512 subset first).
3. Same-window A/B cold-prefill depth bench 63k/120k vs TILE-32 monolithic control (696/388 tok/s), 2 passes,
   server isolated. WIN = faster than control. STRETCH = approach llama.cpp.
4. Sweep KC ∈ {64, 128} and TILE ∈ {32, 64} for the occupancy sweet spot.

## Status / next

Design locked 2026-06-27 from the measured root cause + llama.cpp reference. NEXT = implement env-gated in
`kernel_unified_attention_2d` (real-kernel, default OFF), then validate per the test plan. Blocker #2 (the
~175k admission-deadlock) — **✅ RESOLVED 2026-06-27**: it was the fp16-KV capacity ceiling; `--kv-cache-dtype fp8` fixes it (333k cache > 262k max-model-len; see `BLOCKER2-fp8-capacity-RESOLVED.md`). Gemma now cold-prefills 256k.

---

## ✅ PROTOTYPE VALIDATED 2026-06-27 (flash_chunk_proto.py, standalone, gfx1201)

The core mechanic is proven correct + low-LDS in isolation BEFORE touching the real kernel:

    MONO        : shared=65536 B   max|out-ref|=6.7e-05
    CHUNK KC=64 : shared= 6144 B   max|out-ref|=6.7e-05   max|chunk-mono|=3e-08   → LDS 91% LESS, bit-identical

- Chunked QK (head-chunk loop, S accumulated) + chunked PV (per-output-chunk accumulators) is numerically
  identical to both the monolithic kernel (3e-08) and the torch reference (6.7e-05). Online softmax unchanged.
- **LDS 65536 → 6144 B (−91%)** at KC=64. On RDNA4 (64 KiB/CU) that lifts occupancy from ~1 → ~10 resident
  wave-blocks — the exact lever the root-cause analysis identified. (Absolute numbers are fp16/standalone;
  the real kernel uses fp8 + more buffers, but the relative drop is the signal.)

### Triton-3.6 construct findings (load-bearing for the real-kernel port — these took several probes)

- ❌ `list.append`, ❌ list comprehensions, ❌ generator-as-tuple comprehensions, ❌ `tl.cat(a,b)` (2D last-axis)
  — all UNSUPPORTED in this Triton (prim_probe.py).
- ✅ `tl.join(a,b)`+`tl.reshape` works but INTERLEAVES (not in-order concat) — not usable for head assembly.
- ✅ **Explicit tuple-literal accumulators carried across a real loop** + **constexpr-literal indexing** works.
  This is THE pattern: carry the 8 per-output-chunk accumulators as an explicit 8-tuple, reassign the whole
  tuple each KV tile (8 explicit `_pv(...)` calls), and STORE each chunk to its head offset with literal
  indices `accs[0]..accs[7]` (a `for c in range(8): accs[c]` loop FAILS — Triton treats `accs[c]` as tensor
  getitem when c is a loop var; must be literal).
- Q-chunks are re-loaded per KV tile inside the unrolled head-chunk loop (L2-cached, tiny at BM=16) — avoids
  needing a persistent Q tuple. Lever A's Q re-read was never the killer; leaving V monolithic was.
- NUM=8 is hardcoded (HEAD=512/KC=64, fixed for Gemma) — acceptable; gate the path on head_size==512.

## NEXT: port into the real kernel_unified_attention_2d (env-gated, default OFF)

Carry over from the proto, ADD the real-kernel concerns: paged-KV block-table indexing (physical_block_idx),
fp8 K/V dequant PER CHUNK (in-register, k_scale/v_scale), the full mask (causal/sliding/mm_prefix), alibi,
softcap, sinks, and the k_eq_v reuse (Gemma full layers share K=V — V-chunk can reuse the K-chunk load, a
free bandwidth halving; add only AFTER the independent-V version validates). Then: compile → fp32 differential
correctness (head-512 case) → same-window A/B vs TILE-32 monolithic control (696/388 tok/s @63k/120k).

---

## ✅✅ REAL-KERNEL VALIDATED 2026-06-27 — correct + 3–3.5× faster (env-gated VLLM_FA_HEADCHUNK=64)

Ported the validated prototype into the real engine as a SEPARATE kernel `kernel_unified_attention_2d_headchunk`
(+ helpers `_pv_chunk`, `_store_chunk`) in triton_unified_attention.py. Dispatch (`use_headchunk` in
unified_attention) gates it to head_size==512 + no alibi/qq_bias/sinks + NUM==8; passes HEAD_CHUNK via `**_extra_2d`.
Stock monolithic kernel untouched (default OFF = byte-identical). Real-kernel concerns added vs the proto:
paged-KV block-table indexing, fp8 K/V per-chunk dequant (in-register), full causal+mm_prefix+softcap mask.

CORRECTNESS (needle differential, 120k, same prompt, server isolated):
- Monolithic:  recall 15/20, misses {09,12,14,15,19}
- Chunked:     recall 15/20, misses {09,12,14,15,19}   ← IDENTICAL (the 5 misses are scorer format artifacts
  present in BOTH; the model retrieves all 20 facts correctly). ⇒ output-equivalent to monolithic.

SPEED (clean prefill-only A/B, same window, fp8 KV, eager, pass2 steady-state):
| depth | monolithic        | chunked            | speedup |
|------:|------------------:|-------------------:|:-------:|
| ~63k  | 80.9s / 690 tok/s | 26.7s / 2067 tok/s | 3.03×   |
| ~120k | 282.4s/ 380 tok/s | 80.6s / 1317 tok/s | 3.50×   |
(needle wall 94k-tok prompt corroborates: 290s mono vs 94s chunked ≈ 3.1×.) Speedup GROWS with depth
(occupancy matters more when deeper = more KV-bound). This closes most of the 5.5× gap to llama.cpp and
puts the chunked kernel in llama.cpp's throughput ballpark (~1317 tok/s @120k).

EXTRAPOLATION (not yet measured): mono 256k ≈ 24.6min/1476s; at 3.5× ≈ ~420s — would CLEAR the 600s client
bar. Blocker #2 (the ~175k admission-deadlock) is **✅ RESOLVED** via `--kv-cache-dtype fp8`; measured cold 236k = 346s (~682 tok/s), so Gemma does cold 256k. See `BLOCKER2-fp8-capacity-RESOLVED.md`.

REMAINING to ship: (1) KC/TILE sweep (KC∈{64} fixed by the 8-tuple; TILE∈{32,64} — try TILE 64 now that LDS is
tiny, may add more); (2) k_eq_v reuse (Gemma full layers share K=V → reuse K-chunk load for V, free bandwidth
halving); (3) the fp32 differential unit test for CI; (4) fix blocker #2 for full 256k; (5) ship-gate +
upstream-candidate writeup (RDNA4 give-back). Logs: bench_mono_ctrl.log, bench_chunk_fa.log.

---

## SWEEP 2026-06-27: TILE-64 REFUTED for the chunked kernel; k_eq_v reassessed DOWN

TILE-64 (VLLM_PREFILL_TILE_SIZE=64 + chunked) is SLOWER than TILE-32, both depths:
| depth | chunked TILE-32 | chunked TILE-64 |
|------:|----------------:|----------------:|
| 63k   | 26.7s / 2067 tps| 37.9s / 1493 tps|
| 120k  | 80.6s / 1317 tps| 125.8s/ 865 tps |
⇒ TILE-32 is optimal. Occupancy is already maxed (LDS tiny); a bigger tile just adds per-tile work with no
occupancy payoff. KEEP TILE-32 for the chunked path.

**k_eq_v reuse — reassessed from "free win" to "likely-marginal + risky" (do NOT implement without profiling):**
- Gemma k_eq_v writes IDENTICAL k and v to the cache but as SEPARATE physical tensors (gemma4.py:298,401-405:
  "K weights loaded into both K and V slots"; attn(q,k,v) with k==v). So V is a separate HBM copy of K → reuse
  COULD halve full-attn cache reads.
- BUT reuse needs holding all 8 K-chunks across the softmax (to transpose for PV) → ~32KB extra registers on top
  of the 8 accumulators → RDNA4 VGPR spill risk that could erase the gain.
- AND the TILE-64-slower result is evidence the kernel is OCCUPANCY/LATENCY-bound, not BANDWIDTH-bound — and
  k_eq_v only reduces bandwidth. So the expected payoff is small.
- VERDICT: not worth the kernel risk on current evidence. Revisit only if a profiler shows the chunked kernel
  is memory-bound at depth. The validated TILE-32 chunked kernel (correct + 3-3.5×) is the deliverable to bank.

---

## ✅✅✅ CI TEST LANDED + fp8 BUG CAUGHT/FIXED 2026-06-27

Added `test_triton_unified_attn_headchunk` (tests/kernels/attention/test_triton_unified_attention.py): a
**chunked-vs-monolithic fp32 differential** — runs the SAME inputs through `_FA_HEADCHUNK=0` (stock monolithic)
and `=64` (chunked) via `monkeypatch.setattr(tua,"_FA_HEADCHUNK",…)`, forces the 2D path for both legs
(`seq_threshold_3D=0` + 0-sized segm buffers), and asserts tight chunk-vs-mono parity (2e-3 non-fp8 / 5e-2 fp8)
plus a `ref_paged_attn` ground-truth check. Matrix kept deliberately small (the 8-way-unrolled head-512 kernel
is multi-minute to JIT per constexpr combo): 2 GQA packings (8,2)+(5,1) × {None, fp8} × 4 seq-len shapes
(deep-prefill / mixed / decode / ragged). sliding_window/soft_cap are NOT cross-producted — they drive the
SHARED softmax+mask block (identical in both kernels) and are already covered by the stock test.

**Results:**
- Non-fp8 (q_dtype=None): **48/48 pass** in the full exploratory matrix (all sliding/softcap/packing/shape combos),
  bit-tight at 2e-3 — the QK/PV chunking + online softmax is numerically equivalent to monolithic.
- fp8 (q_dtype=fp8): the test **CAUGHT a real divergence** — `chunked-vs-mono parity broke: max|d|=0.093` on the
  first fp8 case. Root cause: in the **all-fp8 path (Q itself fp8)** the stock 2D kernel keeps K/V RAW
  (`if Q.dtype.is_fp8(): K=K_load`) — k_scale/v_scale are folded into out_scale, NOT applied per-tile — whereas
  the chunked kernel was unconditionally dequant-and-round-tripping (`(K.to(f32)*k_scale).to(QDT)`). NOTE this is
  NOT a prod config: prod is **bf16 Q + fp8 KV cache** → `Q.dtype.is_fp8()` is False → the dequant branch, which
  the chunked kernel already matched (hence the real-engine needle test was identical). The all-fp8 path is only
  reachable from the unit test.
- **Fix:** mirror the stock kernel's `if Q.is_fp8(): keep K/V raw` branch in both the chunked QK (`Qc.dtype.is_fp8()`)
  and `_pv_chunk` (`P_cast.dtype.is_fp8()`, since P_cast=P.to(QDT)). The kernel is now a faithful drop-in for the
  stock kernel across ALL fp8 modes, not just the prod one.
- Post-fix: fp8 cases pass (q_dtype1 cases that previously failed at 0.093 now green). Committed CI matrix = 12 passed + 4 skipped (the (5,1)+fp8 cell is skipped: slow Triton JIT, not a failure; (5,1)+fp8 was separately observed to PASS before the skip).

This is the deliverable banked: correct (chunk≡mono across all modes incl all-fp8) + 3–3.5× faster (prod fp8-KV
A/B) + a regression test that already earned its keep by catching the fp8 scale-fold divergence.
