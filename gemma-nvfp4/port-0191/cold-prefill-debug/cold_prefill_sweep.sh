#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Cold-prefill DEPTH-SCALING discriminator (cold-prefill HANDOFF, Step A).
#
# Fires ONE cold (prefix-cache MISS) request at each of several context depths and
# records the SERVER-SIDE prefill throughput (vLLM's own "Avg prompt throughput:
# N tokens/s" log line) plus GPU utilisation. The slope of prefill tok/s vs depth
# is the decisive, profiler-free discriminator between the two live hypotheses:
#
#   H1 attention-bound (2D triton kernel on the 5 full-attention layers, O(n^2)):
#       prefill tok/s  ∝ 1/depth   → roughly HALVES as depth doubles.
#   H2 MoE-bound (NVFP4 emul kernel, linear in tokens, O(1) in depth):
#       prefill tok/s  ≈ FLAT across depths.
#
# Run ISOLATED (stop any other GPU workloads first) so nothing steals the GPU —
# contamination is exactly what made the original "85 tok/s" untrustworthy.
#
# Usage:
#   ./cold_prefill_sweep.sh                              # gemma, depths 40k/80k/160k
#   MODEL=qwen CONTAINER=<your-vllm-container> ./cold_prefill_sweep.sh   # point at another served model
#   DEPTHS="40000 80000" TIMEOUT=900 ./cold_prefill_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-gemma}"
CONTAINER="${CONTAINER:-vllm-gemma}"
DEPTHS="${DEPTHS:-40000 80000 160000}"
TIMEOUT="${TIMEOUT:-2400}"            # generous: a 160k cold prefill at ~85 tok/s ≈ 30 min
THINK_FLAG="${THINK:+--think}"
OUT="${OUT:-$HERE/sweep-$(date -u +%Y%m%dT%H%M%SZ).log}"

echo "sweep: model=$MODEL container=$CONTAINER depths=[$DEPTHS] timeout=${TIMEOUT}s think=${THINK:-0}" | tee "$OUT"
command -v docker >/dev/null && docker inspect "$CONTAINER" >/dev/null 2>&1 \
  || { echo "WARN: container '$CONTAINER' not found — server-side throughput capture will be skipped" | tee -a "$OUT"; }

declare -a RESULTS
for D in $DEPTHS; do
  echo | tee -a "$OUT"; echo "════════ depth target=${D} tokens ════════" | tee -a "$OUT"
  # mark the docker log boundary so we only scan THIS request's prefill lines
  SINCE="$(date -u +%Y-%m-%dT%H:%M:%S)"
  # background GPU-use sampler (confirms GPU-100% compute, not a host-side stall)
  GPUSAMP="$HERE/.gpusamp.$$"; : > "$GPUSAMP"
  ( while :; do rocm-smi --showuse 2>/dev/null | grep -E 'GPU use' | tr '\n' ' '; echo; sleep 5; done ) >> "$GPUSAMP" 2>/dev/null &
  SAMP_PID=$!

  # the request (server reports prompt_tokens; harness prints COMPLETED/HANG)
  HARNESS_OUT="$(python3 "$HERE/cold_prefill_test.py" \
      --model "$MODEL" --target-tokens "$D" --timeout "$TIMEOUT" $THINK_FLAG 2>&1)"
  echo "$HARNESS_OUT" | tee -a "$OUT"
  kill "$SAMP_PID" 2>/dev/null

  # pull server-side prefill throughput for this window (vLLM's own measurement)
  PT="$(echo "$HARNESS_OUT" | grep -oE 'prompt_tokens=[0-9]+' | grep -oE '[0-9]+' | head -1)"
  WALL="$(echo "$HARNESS_OUT" | grep -oE 'in [0-9]+s' | grep -oE '[0-9]+' | head -1)"
  PREFILL_TPS="n/a"
  if docker inspect "$CONTAINER" >/dev/null 2>&1; then
    PREFILL_TPS="$(docker logs --since "$SINCE" "$CONTAINER" 2>&1 \
        | grep -oE 'Avg prompt throughput: [0-9.]+ tokens/s' \
        | grep -oE '[0-9.]+' | sort -gr | head -1)"
    [ -z "$PREFILL_TPS" ] && PREFILL_TPS="none(see log)"
  fi
  GPUPEAK="$(grep -oE '[0-9]+' "$GPUSAMP" 2>/dev/null | sort -gr | head -1)"; rm -f "$GPUSAMP"
  RESULTS+=("depth=${D} prompt_tokens=${PT:-?} wall=${WALL:-?}s server_prefill_tok/s=${PREFILL_TPS:-?} gpu_peak%=${GPUPEAK:-?}")
  echo ">> ${RESULTS[-1]}" | tee -a "$OUT"
done

echo | tee -a "$OUT"; echo "════════ SUMMARY (slope = the answer) ════════" | tee -a "$OUT"
printf '%s\n' "${RESULTS[@]}" | tee -a "$OUT"
echo "Interpretation: server_prefill_tok/s halving as depth doubles ⇒ H1 attention-bound;" | tee -a "$OUT"
echo "                roughly flat ⇒ H2 MoE-bound. (Use server_prefill_tok/s, not wall.)" | tee -a "$OUT"
echo "full log: $OUT"
