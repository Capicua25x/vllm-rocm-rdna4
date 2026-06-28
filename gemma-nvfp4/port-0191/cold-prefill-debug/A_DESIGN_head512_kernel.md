# Lever A — vLLM RDNA4 head-512 prefill kernel fix (design brief, 2026-06-27)

The three analysis threads came back empty (`THREAD RESULTS: []`), so I grounded this brief by reading the vLLM kernel, the dispatch/tile-gate logic, the fp8-Q gate, the llama.cpp tile-config tables, and the fattn.cu kernel-selection logic directly. The single most decisive finding from that read reshapes the whole plan: **llama.cpp's fast head-512 kernel is NOT a WMMA kernel** — `fattn.cu:447` excludes `Q->ne[0]==512` from the WMMA path and `:454` caps the RDNA4 WMMA path at `Q->ne[0]<=128`, with the comment "AMD WMMA is only faster if the full tile width of 16 can be utilized." Head-512 on RDNA4 falls through to `BEST_FATTN_KERNEL_TILE`, a hand-tiled FMA kernel. That inverts the prompt's option (b).

---

# LEVER A — IMPLEMENTATION DESIGN BRIEF: vLLM head-512 prefill on RDNA4

## 1. ROOT CAUSE (mechanism)

The Triton `kernel_unified_attention_2d` computes the QK dot (`triton_unified_attention.py:350`, `S += scale*tl.dot(Q,K)`) with the **entire** `HEAD_SIZE_PADDED=512` as the contraction held in one `tl.dot`. Triton stages the persistent Q operand (`BLOCK_M=128 × 512 × 2B fp16 = 131072 B = 128 KiB`) into LDS for the matrix-core lowering. RDNA4 has only ~64 KiB LDS/CU, so this **tile-independent** 128 KiB Q buffer means the kernel can only launch at the minimum `TILE_SIZE=32` and ~1 resident wavefront-block — occupancy-starved, no latency hiding. This is confirmed by the known failure mode: `VLLM_PREFILL_TILE_SIZE=64/128` both crash `"shared memory Required 131072 > limit 65536"` — and 131072 is exactly the Q tile, *independent of TILE_SIZE*, proving the Q-LDS residency (not the KV tile) is the cap. The 5 full-attention layers run wave-starved; that is the 5.5x gap. llama.cpp avoids it by never keeping 512 resident: it chunks the head-dim contraction into `nbatch_K=64` slices and KV depth into `nbatch_fa=64` (RDNA table, `fattn-tile.cuh:267-270`), staging only ~64×64 in LDS with a register accumulator and a hand-unrolled FMA inner loop (`ggml_cuda_mad`, `fattn-tile.cuh:511`) — sustaining occupancy 2. It explicitly rejects WMMA at head-512.

## 2. THE FIX — ranked

### Option A — head-dim chunked QK/PV dot (port llama.cpp `nbatch_K`). **BEST BET, and the first real step.**
- **What/where:** In `triton_unified_attention.py`, rewrite the inner tile body of both kernels (2D `:347-414`, 3D `:702-768`). Replace the single `tl.dot(Q,K)` over 512 with a loop over `HEAD_DIM_CHUNK` (start 128, sweep 64/256) slices accumulating into the fp32 `S`; mirror for the PV dot by tiling the 512-wide output free-dim of `acc`. Add a head-512 branch in `_get_tile_size` (`:891`) / the OOM gate so `TILE_SIZE_PREFILL` can rise to 64–128 once LDS is freed. Env-gate it (e.g. `VLLM_HEAD512_CHUNK_K`) so default behavior is unchanged.
- **Technique:** llama.cpp `nbatch_K` head-dim chunking — split-K inside the kernel to decouple LDS from head_dim.
- **Effort:** Medium (kernel inner-loop rewrite + retune the tile/OOM gate + extend the differential test).
- **Expected:** This is the mechanism that makes llama.cpp 5.5x faster. Realistic landing 2–3x over current 167 tok/s if Triton's allocator cooperates; the silicon ceiling (llama.cpp 918–952) leaves ample headroom.
- **Risk (gfx1201 Triton):** MEDIUM-HIGH. Triton may still stage the full Q in LDS or fail to pipeline a manual split-K, partially negating the LDS win; Triton's RDNA4 ceiling is below hand-tuned HIP regardless. **Mitigate by measuring Triton's reported shared-mem alloc after chunking before committing** — if the 128 KiB doesn't drop, A is in trouble and pivot to C.
- **Note:** the cheap "just raise TILE_SIZE" retune is **already disproven** (it OOMs) — that OOM is itself the proof that the Q-tile LDS residency is the root cause, which is why the minimal A is the floor of the work, not an optional retune.

### Option B — native-fp8-Q path (un-gate `triton_attn.py:441`). Second.
- **What/where:** `self.supports_quant_query_input = current_platform.is_cuda()` → also allow ROCm/gfx1201. Keeps Q in fp8 so the QK dot is fp8×fp8: the persistent Q LDS operand halves (`128×512` fp8 = 64 KiB vs 128 KiB), which *directly relieves the exact LDS cap from §1*, and gfx1201 can emit `v_wmma_f32_16x16x16` fp8 (vLLM issue #28649 says Triton *can*).
- **Effort:** Medium (un-gate + confirm Triton emits the fp8 dot on ROCm + validate numerics).
- **Expected:** Could enable TILE_SIZE=64 by itself (LDS relief), independent of A; stacks with A.
- **Risk:** HIGH on two axes — (1) Triton *can* ≠ *does* emit fp8 WMMA on gfx1201 here; (2) fp8-Q in prefill is numerically unvalidated (the `is_cuda()` gate likely exists for accuracy) — must clear the differential + needle test. Don't lead with it, but it is the natural fallback if A's chunking doesn't drop peak LDS.

### Option C — emit f16 WMMA for the attention dots. **DO NOT INVEST (explicit non-recommendation).**
- llama.cpp **benchmarked this and excluded head-512 from WMMA on RDNA4** (`fattn.cu:447` `Q->ne[0]!=512`; `:454` caps WMMA at `<=128`; `:474` "WMMA only faster if the full 16-wide tile is utilized"). The deep head-512 reduction underutilizes the 16-wide WMMA tile and loses to tuned FMA. Confidence HIGH this is a dead end — it's llama.cpp's own measured decision on the same silicon. This is the tempting lever the prompt asked us to weigh; the evidence says skip it.

**Best first bet: Option A** (it is also the first concrete step; the pure-retune shortcut is already OOM-disproven). Keep B in reserve for LDS relief if A's chunking doesn't lower peak shared-mem; skip C.

## 3. THE TARGET

- 247k cold prefill under 600s ⇒ ≥ **412 tok/s**; 256k ⇒ ≥ **427 tok/s**. Current vLLM is 167 tok/s (1475s) — need **~2.5x**.
- llama.cpp does 918–952 tok/s (~270s @ 247k) on the same cards ⇒ the silicon clears 600s with >4 min margin. A 2.5x is conservative against that existence proof, so **A plausibly reaches the compute bar.**
- **Two honest caveats:** (1) **MTP roughly doubles cold prefill** (`AB_RESULT` §157k: 382s MTP vs ~205s eager) — apply/measure the lever on the eager/SPEC-off prefill path, which is what governs the timeout. (2) **A is necessary but not sufficient for 256k**: vLLM 0.19.1/RDNA4 hits an **admission deadlock at ~175k** (`AB_RESULT` "CLIFF FOUND ≈170-180k") — the 5 head-512 full layers exhaust KV cache at gpu-mem 0.92/TP2; the request never starts (GPU 3%, prompt_tokens=0), independent of MTP and the 3D fix. So A makes **≤170k comfortably sub-600s immediately**, and only reaches 256k once the capacity/admission cliff is fixed in parallel (more VRAM headroom or the scheduler-spin bug). State this in the plan: lever A targets the right number on compute; the >175k deadlock is a separate fix — **✅ RESOLVED 2026-06-27** via `--kv-cache-dtype fp8` (333k cache > 262k max-model-len; see `BLOCKER2-fp8-capacity-RESOLVED.md`). Gemma now cold-prefills 256k.

## 4. TEST PLAN

- **Correctness:** Extend `test_triton_unified_attn_3d_large_query` (2D-vs-3D fp32 differential, full matrix incl. `head_dim=512` — already 216/216 per `AB_RESULT`) to exercise the new chunked-dot path; require 0 failures across the head-512 matrix + the full-file regression prefix. Then `needle256k.py` multi-needle recall (20 canaries) at the depth that admits today (~169k), and at 256k once capacity allows — require recall parity with the Qwen reference (Qwen 258k clean).
- **Speed:** A/B depth bench (`cold_prefill_test.py` / the `AB_RESULT` harness), **server isolated** (no competing `:8011` traffic — cross-window variance is large per the "CORRECTIONS" section; the contaminated "85 tok/s" alarm is the cautionary tale), nonce-block-0 cold (`cached=0` verified), legs differing ONLY by the new env flag. Measure tok/s + TTFT at 16k/63k/120k/160k (and 247k if it admits). **Use the already-shipped +30% 3D-fix as the control**, not the pre-fix baseline.
- **Non-determinism caveat:** run **2+ passes per leg**, discard the first (cold) repeat per the methodology note (`AB_RESULT` "always warm up / discard the first bench repeat"), and trust same-window A/B deltas, not absolute single-window numbers (Qwen `top_p=0.95/top_k=20` + cold-prefill cross-window variance).

## 5. FIRST CONCRETE STEP

**Change:** Implement the minimal core of Option A — env-gated (`VLLM_HEAD512_CHUNK_K=1`) head-dim chunking of the QK dot in `kernel_unified_attention_2d` (`triton_unified_attention.py:347-414`): loop the contraction over 128-wide head-dim slices with an fp32 `S` accumulator instead of one `tl.dot` over 512, then allow `TILE_SIZE_PREFILL=64` for the head-512 case in `_get_tile_size`.

**Measure:** (1) First read Triton's reported shared-mem allocation — confirm it drops below the 64 KiB limit (i.e. the 131072→ smaller). That single number tells you whether A's mechanism works on gfx1201 *before* you tune anything. (2) Then run the A/B depth bench at **63k and 120k cold, 2 passes each, server isolated**, comparing against the 3D-fix control.

**GPU required:** YES. This is an empirical LDS/occupancy question — the static OOM math (128 KiB > 64 KiB) tells you the *problem* but only the R9700s can tell you whether the chunked dot actually lowers peak shared-mem and lifts occupancy on the gfx1201 Triton backend. Budget ~1 hour on a side vLLM instance; the shared-mem readout alone de-risks (or kills) the whole lever.

**Confidence:** HIGH on the root cause (the OOM message + llama.cpp config tables + the WMMA exclusion are unambiguous). MEDIUM on A's magnitude (Triton allocator cooperation on gfx1201 is the open variable). HIGH that C (WMMA) is a dead end. HIGH that the 256k goal additionally requires fixing the separate ~175k admission cliff.

---

Files referenced (all absolute):
- vLLM kernel: `<vllm-src>/vllm/v1/attention/ops/triton_unified_attention.py` (2D `:88-432`, 3D `:435-788`, dispatch+gates `:918-1192`, `_get_tile_size` `:891-915`)
- vLLM fp8-Q gate: `<vllm-src>/vllm/v1/attention/backends/triton_attn.py:441`
- llama.cpp tile configs (RDNA head-512): `<llama.cpp-src>/ggml/src/ggml-cuda/fattn-tile.cuh:267-270` (config), `:448-517` (chunked FMA inner loop)
- llama.cpp WMMA exclusion of head-512 on RDNA4: `<llama.cpp-src>/ggml/src/ggml-cuda/fattn.cu:447,454,474`
- Prior A/B measurements + the ~175k admission cliff: `gemma-nvfp4/port-0191/cold-prefill-debug/AB_RESULT_3dprefill.md`

External tools used this session: none — all reads via local `Read`/`grep`/`sed`. No API-side tools.

---

## GO/NO-GO RESULT (2026-06-27) — PASSED

Standalone Triton probe `head512_lds_probe.py` (in-container, gfx1201), head-512 attention tile,
monolithic vs head-dim-chunked QK **load**:

| Mode | TILE 32 | TILE 64 | TILE 128 |
|---|---|---|---|
| Monolithic   | 131072 ❌ | 131072 ❌ | 131072 ❌ |
| Chunked(128) | 32768 ✅  | 65536 ✅  | 131072 ❌ |
| Chunked(64)  | —         | 65536 ✅  | 131072 ❌ |

- Monolithic LDS = 131072 B at EVERY tile = the tile-independent 128×512 fp16 Q operand (brief confirmed).
- Chunked-LOAD (load Q/K in head-dim slices, never staging a full-512 operand) drops LDS to 32 KiB and
  fits TILE 64 within RDNA4's 64 KiB. The "does Triton's gfx1201 allocator cooperate" unknown = YES.
- TILE 128 still OOMs (the monolithic PV V operand TILE×512 binds at 64) — TILE 64 is the win for now;
  chunk PV later for >64.

## IMPLEMENTATION PLAN (real kernel)

Files: `triton_unified_attention.py` (2D kernel + mirror in 3D), `triton_attn.py` dispatch.
1. Add `HEAD_DIM_CHUNK: tl.constexpr` (0 = off, default) to both kernels.
2. 2D kernel: when chunk>0, SKIP the full Q load (`:177`); compute `S` via `tl.static_range` over head-dim
   chunks, loading Q-chunk (`query_ptr`) + K-chunk (`key_cache_ptr`) per chunk, applying the fp8 K dequant
   (`:291`) per chunk, accumulating into `S` (then `*scale`). Preserve a query-dtype handle for the fp8 check.
3. `_get_tile_size` (`:891`): allow `TILE_SIZE_PREFILL=64` for head-512 when chunk enabled.
4. Dispatch (`:927-1192`): read `VLLM_HEAD512_CHUNK`, set `HEAD_DIM_CHUNK=128` for head_size==512 (else 0),
   pass to both 2D + 3D launches.
5. Validate: vLLM compile (no OOM @ TILE 64) → differential test (correctness) → A/B depth bench 63k/120k
   vs the +30% 3D-fix CONTROL, 2 passes, server isolated. Risk to measure: chunked-load re-reads Q per KV-tile
   (bandwidth) — net speedup is empirical. Backup: `_backups/triton_unified_attention.py.pre-head512chunk`.

---

## ❌ RESULT 2026-06-27 — LEVER A REFUTED (net SLOWER, ~1.8–2×). Negative result.

Implemented end-to-end (env `VLLM_HEAD512_CHUNK=128`): head-dim chunked QK in `kernel_unified_attention_2d`
+ `_get_tile_size` returns TILE 64 for head-512 prefill when chunk>0 + dispatch wiring (`head_dim_chunk`).
Compiles at TILE 64 — the `131072 > 65536` LDS OOM that blocked any tile >32 is GONE — and is numerically
correct (smoke 17×23=391). So the LDS go/no-go held. **But throughput got WORSE.**

Same-window A/B, Gemma-4-26B-A4B-NVFP4, fp8 KV, eager, SPEC=0, MODEL_LEN=262144, MAXSEQS=8, 2D path
(NO VLLM_TRITON_3D_PREFILL — isolates the 2D head-512 kernel). Cold prefill = unique nonce per request
(prefix-cache miss). Steady-state = pass2 (pass1 includes first-run kernel compile):

| depth | TREATMENT chunk128 / TILE-64 | CONTROL monolithic / TILE-32 | control advantage |
|------:|-----------------------------:|-----------------------------:|:-----------------:|
| ~63k  | 147.2s — 380 tok/s           | 77.3s — 696 tok/s            | 1.83× faster      |
| ~120k | 525.9s — 205 tok/s           | 266.8s — 388 tok/s           | 1.97× faster      |

WHY IT LOST (both costs were flagged as risks pre-bench; both bit):
1. **Q re-read from HBM per KV tile.** Monolithic loads Q (128×512) ONCE before the KV-tile loop and reuses
   it across all tiles. The chunked path loads Q in head slices INSIDE the QK section, i.e. re-reads the
   whole Q tile every KV tile (~N_tiles× the Q HBM traffic). TILE-64 only halves N_tiles; it cannot offset
   re-reading Q hundreds/thousands of times.
2. **MFU loss.** Four small (128×128)@(128×64) dots utilize the RDNA4 matrix units worse than one
   (128×512)@(512×32) dot. The TILE-64 occupancy headroom does not pay for 1+2.

WHY THE OBVIOUS FIX (hoist Q-chunk loads out of the KV loop = "Q resident") IS NOT VIABLE HERE:
- Keeping the full Q resident across the loop = 128KB. As 4×(128×128) register-held chunks that's ~1024
  VGPRs/thread → massive spill on RDNA4 (≤256 VGPR). As one LDS-staged (128×512) tensor it's the original
  128KB LDS that pins TILE at 32. Either way you're back to the starting constraint.
- **BLOCK_M is pinned to the GQA group (=128).** Gemma-4 full layers map ~128 query heads onto 1 KV head/rank,
  so `BLOCK_M = next_pow2(num_queries_per_kv) = 128`, `BLOCK_Q = BLOCK_M//num_queries_per_kv = 1`. Forcing
  BLOCK_M=64 makes BLOCK_Q=0 and breaks tiling. Shrinking Q therefore needs splitting the GQA group across
  programs — invasive kernel surgery, uncertain payoff. That is the ONLY remaining in-kernel lever, and it is
  NOT a quick patch; it's a rewrite of the head-grouping in the grid (≈ porting llama.cpp's hand-tiled FA).

STATUS OF THE CODE: edits kept env-gated, default OFF (`VLLM_HEAD512_CHUNK` unset = byte-identical to stock
behavior), so the tree is safe and the negative result is reproducible. Full revert if desired:
`cp _backups/triton_unified_attention.py.pre-head512chunk <tree>/v1/attention/ops/triton_unified_attention.py`.

BOTTOM LINE: the simple/cheap kernel lever (chunk QK → bigger TILE) is dead. Best IN-vLLM result remains the
already-validated `VLLM_TRITON_3D_PREFILL=1` KV-split (+28–32%, 120k≈807 tok/s in its window). Reaching the
llama.cpp bar (247k in ~4.5min) needs either the invasive GQA-split/FA rewrite or serving Gemma via llama.cpp.
