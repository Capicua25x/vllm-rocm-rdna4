#!/bin/bash
# Example LLM serving wrapper — DeepSeek-V4-Pro-Thinking DISTILL of Qwen 3.6-35B-A3B
# (MXFP4 + Vision) on vLLM 0.19.1 (RDNA4), TP2 across both Radeon AI PRO R9700s. Serves :8011.
#
# This wraps the BAKED image  capicua25x/vllm-rocm-rdna4:0.19.1  (built on Rob's frozen 0.18.1 base
# image tcclaviger/vllm-rocm-mxfp4-nvfp4:latest) — vLLM 0.19.1 + the RDNA4 MXFP4 graft + the
# monolithic-TRITON MoE OOB fix + transformers 5.8.1. The image bakes ALL setup (editable install +
# triton_kernels graft + librccl symlink), so this wrapper does NO runtime setup — which lets it run
# with the FULL cap-drop=ALL hardening.
#
# Container hardening (carried over verbatim from the 0.18.1 wrapper, validated 2026-06-21):
#   --cap-drop=ALL + DAC_READ_SEARCH : HF weight files are mode-600 owned by the host user; the container
#       runs as root and cap-drop=ALL removes CAP_DAC_OVERRIDE → DAC_READ_SEARCH (read-only DAC
#       bypass) is REQUIRED or the TP worker dies FileNotFoundError on model-00001-of-*.safetensors.
#   --cap-add=IPC_LOCK + --ipc=host : TP2 workers.   --security-opt=no-new-privileges : block escal.
#   SECCOMP : Docker default profile.
#
# ⚠ LOAD-BEARING: --attention-backend TRITON_ATTN is REQUIRED (added 2026-06-24). vLLM 0.19.1 made
#   ROCM_ATTN the DEFAULT attention backend, but ROCM_ATTN collapses spec-decode (MTP) throughput at
#   concurrency on gfx1201: 6K-prompt @32 concurrent = 5.1 tok/s (ceiling ~1 user) vs TRITON_ATTN 19.4
#   tok/s (ceiling ~16, matching Rob's 0.18.1 20.9 AND beating its single-stream 92 vs 82). The env var
#   VLLM_ATTENTION_BACKEND is NOT read in 0.19.1 — you MUST use the --attention-backend CLI arg.
#   AITER unified (ROCM_AITER_UNIFIED_ATTN) is reportedly even better but needs a newer AITER / CDNA-gate
#   relax (ImportError on aiter.ops.triton.unified_attention as of this image) — future optimization.
# The model + arg set are IDENTICAL to the 0.18.1 wrapper (max-model-len 262144, MTP num_spec=3,
# cudagraph ON / no --enforce-eager, qwen3_xml tool parser + qwen3 reasoning parser). Only the model
# is passed via --model because the baked image ENTRYPOINT is the api_server module (not `vllm serve`).
#
# ROLLBACK: run Rob's 0.18.1 base image (tcclaviger/vllm-rocm-mxfp4-nvfp4:latest) instead of this one.
set -euo pipefail
# ROCm device access: mount the kernel-fusion device + the whole DRI tree — works on any host
# (no hardcoded PCI paths). To scope to specific GPUs, set HIP_VISIBLE_DEVICES on the container.
for d in /dev/kfd /dev/dri; do
  [ -e "$d" ] || { echo "required device $d not found" >&2; exit 1; }
done
exec docker run --rm --name vllm-rocm-rdna4 --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render --ipc=host \
  --security-opt=no-new-privileges --cap-drop=ALL --cap-add=DAC_READ_SEARCH --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e HF_HUB_OFFLINE=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  capicua25x/vllm-rocm-rdna4:0.19.1 \
  --model Capicua25x/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4-Vision \
  --served-model-name qwen --port 8011 --trust-remote-code \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 262144 \
  --attention-backend TRITON_ATTN \
  --enable-prefix-caching --max-num-seqs 64 --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
