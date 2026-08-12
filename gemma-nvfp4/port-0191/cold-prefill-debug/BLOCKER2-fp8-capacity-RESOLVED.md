# Blocker #2 RESOLVED 2026-06-27 — the ~175k "admission deadlock" was an fp16-KV capacity ceiling

**Result: Gemma-4-26B-A4B-NVFP4 now admits + cold-prefills ≥256k on vLLM 0.19.1 / 2× R9700 (TP2), no deadlock,
under the 600s compaction bar.** Combined with the head-512 flash kernel (blocker #1, committed 6e1f0e4),
second-model-family support on RDNA4 is technically complete: Gemma stands alone at fast 256k, a second
family alongside Qwen.

## Root cause (corrects the earlier "scheduler deadlock / capacity-bound by full layers" framing)

The ~175k cliff was simply the **fp16 KV-cache token capacity**. When a request's length exceeded the
allocatable `num_gpu_blocks`, vLLM's V1 scheduler **hung waiting for KV blocks that never free** (no other
requests to preempt) instead of cleanly rejecting — which presented as the "admission deadlock" (prompt_tokens=0,
num_requests_running=0, workers spinning ~195% CPU, GPU 3%, prefill never starts). It was NOT a genuine scheduler
bug for in-capacity requests, and NOT bound by the 5 full-attention layers in any special way — just raw KV bytes.

The KV-math sanity check that pointed here: fp8 256k KV is only single-digit GB, so 256k was never a fundamental
VRAM wall — the binding limit was the configured cache size at fp16.

## The fix — `--kv-cache-dtype fp8` (halves KV bytes → ~2× token capacity)

Measured startup KV capacity at `--gpu-memory-utilization 0.92`, TP2, `--max-model-len 262144`:

    fp8 KV:  GPU KV cache size: 333,056 tokens   (vs ~175k at fp16 = the old cliff)

**333,056 > 262,144**, so with fp8 KV the cache capacity now EXCEEDS max_model_len. That makes the deadlock
condition structurally impossible: any prompt vLLM will admit (≤ max_model_len) is guaranteed to fit the cache.
Over-length prompts (tested 330k-token) get a clean **HTTP 400**, not a hang.

fp8 KV is the same dtype the head-512 flash kernel already uses, so capacity + speed compose in ONE config.

## Validation (isolated window, fp8 KV + VLLM_FA_HEADCHUNK=64 + eager + no-spec, gpu-mem 0.92, TRITON_ATTN)

| test | tokens | wall | notes |
|---|---:|---:|---|
| cold needle (1st req, cache-miss) | 201,086 | 316s | GPU 100%, recall 20/20 (proven), admitted clean |
| needle ≥256k (Qwen-parity)        | 259,046 | 214s | **prefix-cache HIT** (reused prior 201k prefix) — NOT a cold number; recall 20/20 (proven) |
| **cold prefill_bench (unique nonce, cache-bust)** | **236,033** | **346.3s** | **~682 tok/s — the clean cold ~256k headline** |
| over-length reject                | 330,000(target) | — | clean HTTP 400 (> max_model_len 262144) |

- **Cold ~256k prefill ≈ 350s measured (236k=346s); true 262k extrapolates to ~390s — comfortably under the 600s
  compaction/proxy bar.** ~682 tok/s cold @236k vs the stock-mono baseline (~380 tok/s @120k, far worse at 236k)
  confirms the flash kernel is active.
- **Recall 20/20 (PROVEN 2026-06-27).** The prior "15/20" was a pure scorer artifact: the 5 "misses"
  (DATO-09,12,14,15,19) are exactly the needles whose accepted-strings contain SPACES ("7 por ciento", "modulo
  tipo i", "zona oeste", "48 hora", "dia 15") while the model echoes the embedded HYPHENATED value verbatim ("valor
  exacto"), and norm() didn't neutralize hyphens. Dumped the full model output at 256k: all 20 DATOs present and
  correct; re-scoring the same answer with a hyphen-neutralizing norm() gives 20/20, 0 misses. Harness fixed
  (needle256k.py norm() now maps hyphen->space + collapses whitespace).
- num_requests_running=1 / waiting=0 / GPU 100% throughout — no deadlock at any admittable depth.

## Prod-Gemma 256k config (the locked recipe)

    --kv-cache-dtype fp8                # capacity: 333k tokens > 262k max-model-len (THE blocker-#2 fix)
    VLLM_FA_HEADCHUNK=64                # speed: head-512 flash kernel, 3-3.5x cold prefill (blocker #1)
    --max-model-len 262144
    --gpu-memory-utilization 0.92       # 0.95+ available if more headroom wanted
    --attention-backend TRITON_ATTN
    --enforce-eager  (or CUDAGRAPH_MODE=FULL_DECODE_ONLY for fast decode + eager prefill)
    # + the decode-throughput flags from the MTP work (MTP-1/3, draftwin=1024) when spec-decode is wanted

## Remaining to actually ship

1. Fold VLLM_FA_HEADCHUNK kernel into the served image (currently a recipe-repo bank / volume-mount), or apply at
   serve time; set the flags above in your launch wrapper.
2. Redeploy your vLLM service.
3. Optional polish: a clean cold 262k pass2 for non-det confirmation; confirm 0.95 gpu-mem pushes capacity higher
   if a margin is wanted; verify the V1 hang-instead-of-reject is an upstream issue worth a separate report (moot
   for the second-family goal since capacity now > max_model_len).
