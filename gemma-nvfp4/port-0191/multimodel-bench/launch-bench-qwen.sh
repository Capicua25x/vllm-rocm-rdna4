#!/bin/bash
# Standalone bench launcher for the Qwen-family models (base + distill) on :8011.
# COMPILED (no --enforce-eager). Served under alias 'qwen' so one bench harness compares all models.
#   VARIANT=base    -> pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4 (MTP-3)
#   VARIANT=distill -> Capicua25x/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4-Vision (MTP-3, trust-remote-code)
# Both Qwen variants natively support MTP-3 spec-decode (output-lossless, ~2.8x single-user decode).
# Env overrides: HF_CACHE (default $HOME/.cache/huggingface), BENCH_IMAGE (vLLM image tag).
set -euo pipefail
VARIANT="${VARIANT:-base}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
BENCH_IMAGE="${BENCH_IMAGE:-tcclaviger/vllm-rocm-mxfp4-nvfp4:latest}"
for d in /dev/kfd /dev/dri; do [ -e "$d" ] || { echo "missing $d" >&2; exit 1; }; done

if [ "$VARIANT" = "distill" ]; then
  MODEL="Capicua25x/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4-Vision"
  EXTRA=(--trust-remote-code --speculative-config '{"method":"mtp","num_speculative_tokens":3}')
else
  MODEL="pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4"
  EXTRA=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')  # base has MTP-3 too
fi

docker rm -f vllm-bench >/dev/null 2>&1 || true
exec docker run -d --rm --name vllm-bench --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render --ipc=host --security-opt seccomp=unconfined \
  -e HF_HUB_OFFLINE=1 \
  -v "$HF_CACHE":/root/.cache/huggingface \
  "$BENCH_IMAGE" \
  "$MODEL" \
  --served-model-name qwen --port 8011 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 262144 \
  --enable-prefix-caching --max-num-seqs 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  "${EXTRA[@]}"
