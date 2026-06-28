#!/bin/bash
# GPU test — Gemma-4 26B-A4B NVFP4 MoE on RDNA4. TWO patches bind-mounted (no rebuild):
#   1) rdna_nvfp4_moe.patched.py — accept RoutingMethodType.Custom + compute Gemma routing
#   2) gemma4.patched.py        — skip unused W4A4 per-expert input_global_scale (weight-only kernel)
# PREREQS: your vLLM service stopped + 26B downloaded.  Restore after: restart your vLLM service.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
exec docker run --rm --name vllm-gemma26-patched --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render --ipc=host \
  --security-opt=no-new-privileges --cap-drop=ALL --cap-add=DAC_READ_SEARCH --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e HF_HUB_OFFLINE=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v "$D/rdna_nvfp4_moe.patched.py":"/app/vllm/vllm/model_executor/layers/fused_moe/experts/rdna_nvfp4_moe.py":ro \
  -v "$D/gemma4.patched.py":"/app/vllm/vllm/model_executor/models/gemma4.py":ro \
  tcclaviger/vllm-rocm-mxfp4-nvfp4:latest \
  RedHatAI/gemma-4-26B-A4B-it-NVFP4 \
  --served-model-name gemma --port 8011 --trust-remote-code \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 32768 \
  --enable-prefix-caching --max-num-seqs 16
