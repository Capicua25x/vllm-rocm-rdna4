#!/bin/bash
# Gemma-4 26B-A4B NVFP4 MoE on RDNA4 + SPECULATIVE DECODING via Google's assistant draft model
# (google/gemma-4-26B-A4B-it-assistant, 419M). Two patches bind-mounted + draft spec-config.
# max-model-len lowered to 16384 to leave VRAM for the draft model + its KV.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
R1=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render); C1=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-card)
R2=$(readlink -f /dev/dri/by-path/pci-0000:06:00.0-render); C2=$(readlink -f /dev/dri/by-path/pci-0000:06:00.0-card)
exec docker run --rm --name vllm-gemma26-ngram --network=host \
  --device=/dev/kfd --device="$R1" --device="$C1" --device="$R2" --device="$C2" \
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
  --enable-prefix-caching --max-num-seqs 16 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
