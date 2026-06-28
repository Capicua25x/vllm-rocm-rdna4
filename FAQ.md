# FAQ — RDNA4 prefill kernel & serving fixes for vLLM 0.19.1

This repository is an **AMD RDNA4 (gfx1201) enablement recipe for vLLM 0.19.1**. Its **primary goal
was to make Gemma-4 (NVFP4 MoE) a first-class RDNA4 model** — a second, independent model family on the
same RDNA4 image as the Qwen MXFP4 path — which required a new drop-in **head-dim-512 prefill attention
kernel** (Gemma's full-attention layers are head_dim-512) plus an **fp8-KV capacity fix** to reach fast
cold 256k. **Secondary payoff — helps the Qwen3.6-35B-A3B distill too, not just Gemma:** `TRITON_ATTN`
recovers 6K-prompt **concurrency** (≈5.1 → 19.4 tok/s at 32 users), and the MTP **3D spec-verify gate**
(`VLLM_TRITON_3D_SMALL_QLEN`) is a **single-stream 6K-latency win — +15% on the Qwen distill** (compiled,
80.8 → 93.3 tok/s; +114% on Gemma, 32.5 → 69.5), output-lossless and neutral at high concurrency.

It is published for transparency and RDNA4-ecosystem enablement, and doubles as the standing
change-notice for any binary/image that ships these modifications (see *Licensing* below). It is **not**
a claim that vLLM is universally faster, and **not** a request to be merged — it helps on a specific GPU
family, is default-OFF, and ships with a differential test you can run yourself.

---

## What's in here?

Four independent pieces — keep them distinct:

| Piece | What it is | Type |
|---|---|---|
| **Head-512 flash kernel** (`VLLM_FA_HEADCHUNK=64`) | A Triton kernel that streams the 512-deep head dimension in 64-wide chunks for both the QK and PV steps, so the full tile never overflows RDNA4's 64 KiB LDS. | new code, RDNA4-specific |
| **fp8 KV cache** (`--kv-cache-dtype fp8`) | Using an already-supported dtype to raise KV capacity past max-model-len. | config only |
| **TRITON_ATTN backend** (`--attention-backend TRITON_ATTN`) | 0.19.1 defaults RDNA4 to `ROCM_ATTN`, which collapses under concurrent spec-decode; this restores throughput. | config / finding |
| **MTP spec-decode backport + 3D verify gate** | Gemma-4's `gemma4_assistant` draft-model spec decode **backported from vLLM 0.23.0** onto the 0.19.1 tree (2 new files + 5 framework patches), plus a gate that routes the multi-token verify forward onto the 3D split-KV kernel — **+15% Qwen / +114% Gemma single-stream @6K**, output-lossless. | backport + new code |

> There are **three** similarly-named env flags. Don't conflate them:
> `VLLM_FA_HEADCHUNK` = the kernel (the big prefill win) · `VLLM_TRITON_3D_PREFILL` = an
> earlier, smaller prefill lever (~+30%) · `VLLM_TRITON_3D_SMALL_QLEN` = the spec-verify fix.

---

## Why vLLM 0.19.1 and not the latest?

Because **0.19.1 is the last vLLM that runs on torch 2.10.** vLLM 0.20.0 moved to torch 2.11, and
everything newer pins it. The only working RDNA4 (gfx1201) base available is on torch 2.10 / ROCm
7.2.1, and there's no torch-2.11 gfx1201 base yet — AMD's own official `gfx120X` image is also frozen
at torch 2.10 / vLLM 0.19.1. So 0.19.1 isn't a cautious choice; it's the newest version this hardware
stack can actually run.

Tried and failed: a vanilla 0.23 build on torch 2.10 dies at the pip build backend (before the
C-extension compile even starts), and several of the files this recipe patches were refactored away by
0.23. When a torch-2.11 gfx1201 base appears, moving forward becomes possible.

---

## Is this a universal speedup? Will it make my GPU faster?

**Almost certainly not, unless you run a head-dim-512 model on RDNA4.** The kernel only engages
when *all* of these hold: the env var is set, `head_size == 512`, no alibi/qq_bias/sinks, and
`512 / chunk == 8`. With the env var unset (the default), the **stock kernel runs, byte-identical.**
On other GPU architectures it's untested and almost certainly irrelevant. This targets one
specific bottleneck on one specific GPU family.

---

## How much faster, exactly — and how was it measured?

On RDNA4 (dual R9700, TP2), cold prefill of head-512 layers, **same-window A/B** (baseline and
fix back-to-back in the same process, only the env var differing):

- 63k tokens: **696 → 2067 tok/s (3.03×)**
- 120k tokens: **388 → 1317 tok/s (3.50×)** — the gain grows with depth.
- LDS per program: **65,536 → 6,144 bytes (−91%)**; occupancy ~1 → ~10 resident wave-blocks/CU.
  *(Those two are prototype-kernel figures; the production kernel's added buffers — paged-KV, fp8
  dequant, full mask — raise absolute LDS. The **speedup** above is the validated real-kernel result.)*

**Important honesty caveat:** absolute cold-prefill throughput on this hardware has high
**cross-window variance** (~2.8× between measurement windows — a known R9700 behaviour, ROCm
issue #6347). **Trust the same-window A/B deltas, not the absolute tok/s.** All figures above are
same-window. A full max-context (262k) figure is *extrapolated* (~390 s, under a 600 s bar) and
corroborated by a *measured* fp8-KV cold point of 236k tokens in 346 s — it is not a measured
same-window A/B at 262k.

---

## Is it correct? How do I know it's not silently wrong?

Don't take it on trust — **run the test.** It ships as a patch
(`gemma-nvfp4/port-0191/head512-flash-prefill.test.patch`); apply it to your vLLM source tree, then run
`tests/kernels/attention/test_triton_unified_attention.py::test_triton_unified_attn_headchunk`:

- Runs identical inputs through the chunked kernel and the stock monolithic kernel (forced via
  monkeypatch so a dispatch bug can't hide), captures output in **fp32** so bf16 rounding can't
  mask a gap, and cross-checks against a pure-PyTorch reference.
- Tolerances: 2e-3 (non-fp8) / 5e-2 (fp8) vs. stock; looser vs. the torch reference.
- Matrix: 4 sequence shapes × 2 GQA packings × {bf16, fp8} (one pathological fp8 cell skipped for
  JIT-compile time; separately observed to pass).

The math is unchanged by design: chunking touches only the *memory layout* of the matmul
operands. The online-softmax / masking block is byte-identical to stock.

---

## Was this AI-assisted? Some of it looks "vibe-coded."

Parts were drafted with AI assistance — and that's exactly why correctness is established by a
**differential test against the stock kernel and a torch reference**, not by trust. The test isn't
decoration: during development it caught a real divergence in the all-fp8 path (the chunked kernel
applied the K/V scale per-chunk where stock keeps K/V raw and folds the scale into the output;
max |Δ| = 0.093). That was found and fixed *because* of the test. If you doubt any line of the
kernel, read it and run the test — the claim rests on reproducible verification, not authorship.

---

## Why not just use a bigger tile, or some simpler fix?

Those were tried and measured (RDNA4, 63k cold):

- **Bigger tile (TILE-64):** 1.87× *slower* than TILE-32 — confirms the bottleneck is occupancy,
  not compute; fatter tiles reduce occupancy further. (At fp16 it doesn't even compile; it only
  compiles with fp8 KV.)
- **Shrink the query tile via GQA-splitting:** rests on a false premise — `BLOCK_M` here is 16, and
  the LDS cap is set by the K/V tiles, which are independent of `BLOCK_M`.
- **Chunk only QK, leave V monolithic:** 1.83× *slower* — re-reads Q each tile, and leaving V
  monolithic keeps LDS high so there's no occupancy gain. **Both** QK and PV must chunk.

The full chunk-both rewrite is the only approach that drops LDS and lifts occupancy.

---

## What are the limitations? (stated up front)

- `NUM=8` (= 512/64) is **hardcoded** — a Triton constexpr-index constraint on the accumulator
  tuple. Other head sizes / chunk widths need a retemplate.
- The 64-wide chunk width was chosen to match llama.cpp's proven `nbatch_K=64`; it wasn't
  exhaustively swept against 32/128.
- `k_eq_v` operand reuse (Gemma writes identical K and V) is **not** implemented.
- Benchmarks are **single-run** cells; expect run-to-run noise.
- Validated on **RDNA4 / gfx1201 only.** Please report results on other hardware.
- The MTP-3 backport runs with **compiled decode + eager prefill** (decode-only cudagraph), as benched;
  a **fully-compiled** (prefill + decode cudagraph) MTP path is the open item.
- 256k kernel throughput is **extrapolated** (measured same-window to 120k).
- **Gemma-4 long context:** with the locked config (`--kv-cache-dtype fp8` + `VLLM_FA_HEADCHUNK=64`),
  Gemma admits + cold-prefills **≥256k** under the 600s bar (cold 236k = 346s; recall 20/20 at 201k;
  over-length → clean HTTP 400). An earlier "~175k admission deadlock" was an fp16-KV capacity limit,
  resolved by fp8-KV (333k cache > 262k max-model-len). Real caveats: cold prefill ~3× slower than
  Qwen (still <600s); MTP doubles cold prefill (use eager/draftwin for compaction). See
  `gemma-nvfp4/port-0191/cold-prefill-debug/BLOCKER2-fp8-capacity-RESOLVED.md`.

---

## Licensing

This code is a modification of vLLM, which is **Apache License 2.0**; these changes are released
under the same license. Changes are delivered as **patch files** (diffs against the stated upstream
vLLM versions) and as **grafted source files that preserve the upstream copyright/SPDX headers**; the
patches plus this repository are the record of what changed. Some files are **backported from vLLM
0.23.0** (the `gemma4_assistant` MTP draft) — same Apache-2.0 upstream, just a newer revision than the
0.19.1 base.

**If you redistribute** (ship the code, a binary, or a container to a third party — including
publishing an image, or installing it on someone else's hardware): include a copy of the Apache
2.0 license, keep the existing copyright/SPDX notices intact, and carry forward this change record
(the patches and `NOTICE`) so the "state your changes" requirement stays satisfied downstream. Apache
2.0 does **not** require you to publish your source — only to preserve notices and state that you
changed things. Merely *running* it (even as a hosted service) is not
redistribution and triggers nothing.

**Models** are a separate matter but, for the stacks this targets, an easy one: Qwen 3.6 and Gemma 4
are themselves **Apache 2.0**, so serving or redistributing those weights carries the same
permissive, notice-only terms — no use-restriction pass-through, no fee, no user cap. Always
confirm the license on the specific model card you actually deploy.

**This recipe builds on Rob Smith's RDNA4 work.** The gfx1201 kernel enablement forward-ported here —
complete hipBLASLt plus the MXFP4/NVFP4 MoE kernels — derives from Rob (`tcclaviger`, image
`tcclaviger/vllm-rocm-mxfp4-nvfp4`), who cracked it for vLLM 0.18.1. Without his base, none of this
exists; full credit is at the top of the README.

The head-chunking approach is a clean re-implementation of a well-known idea (per-64 head-dim
chunking, as used in llama.cpp's `nbatch_K`); credit to that project for the blueprint. No code was
copied — algorithms aren't copyrightable — but the acknowledgement is deserved.

---

## Will you maintain / support this?

Best-effort, on niche hardware, on no schedule. Issues and especially **other-hardware validation
reports** are welcome, but please treat this as a reference recipe you can run and adapt, not a
supported product. Because it's default-OFF, env-gated, byte-identical-to-stock when unset, and
carries its own differential test, it's safe to ignore if it's not for you and safe to try if it is.

---

## How do I try it?

Set the env var and pick the backend at launch:

```
VLLM_FA_HEADCHUNK=64 \
  vllm serve <model> \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype fp8 \
  ...
```

Then apply the test patch to your vLLM tree and run the differential test to confirm correctness on
your box:

```
git apply gemma-nvfp4/port-0191/head512-flash-prefill.test.patch
pytest tests/kernels/attention/test_triton_unified_attention.py -k headchunk
```

With `VLLM_FA_HEADCHUNK` unset, you get stock vLLM, unchanged.
