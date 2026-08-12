#!/bin/bash
# Bench launcher for Gemma-4-26B-A4B-NVFP4 on :8011 — served under alias 'qwen' (and 'gemma')
# so the same bench harness compares it against the Qwen-family models.
#
# This is the FULL-CAPABILITY 256k config (the locked recipe):
#   VLLM_FA_HEADCHUNK=64        head-512 flash-prefill kernel (blocker #1 fix; ~3.5x cold prefill)
#   KV_CACHE_DTYPE=fp8          KV capacity ~304-333k tokens > 262k max-model-len (blocker #2 fix)
#   CUDAGRAPH_MODE=FULL_DECODE_ONLY + ENFORCE_EAGER=0
#                               compiled decode (fast) + eager prefill (sidesteps the compiled
#                               cold-prefill hang) — i.e. "compiled" without the prefill regression
#   SPEC=1 SPEC_TOKENS=3        gemma4_assistant MTP-3 spec-decode draft
#   MAXSEQS=64, GPU_MEM=0.92, MODEL_LEN=262144
#
# Delegates to the patched-tree serve script (set GEMMA_LAUNCHER to override its location).
# That serve script mounts the patched vLLM 0.19.1 tree (RDNA4 NVFP4 grafts + gemma4 MTP backport
# + the head-512 kernel); see ../cold-prefill-debug/ for it and BLOCKER2-fp8-capacity-RESOLVED.md
# for the capacity analysis.
set -euo pipefail
GEMMA_LAUNCHER="${GEMMA_LAUNCHER:-$(cd "$(dirname "$0")/../cold-prefill-debug" && pwd)/launch-gemma-debug.sh}"
[ -x "$GEMMA_LAUNCHER" ] || { echo "Gemma serve launcher not found/executable: $GEMMA_LAUNCHER" >&2; exit 1; }
exec env \
  ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_DECODE_ONLY \
  VLLM_FA_HEADCHUNK=64 KV_CACHE_DTYPE=fp8 \
  MODEL_LEN=262144 GPU_MEM=0.92 MAXSEQS=64 \
  SPEC=1 SPEC_TOKENS=3 \
  SERVED_NAMES="gemma qwen" \
  "$GEMMA_LAUNCHER"
