#!/bin/bash
# Side instance of the Qwen serving image to A/B test the 3D-verify spec-decode fix.
# Mirrors your production launch (same image/model/MTP-3 spec/TP2/parsers) but:
#   * mounts the patched triton_unified_attention.py (env-gated 3D fix) over the image copy
#   * --enforce-eager for a clean fast A/B (prod runs compiled; eager isolates the kernel change)
#   * VLLM_TRITON_3D_SMALL_QLEN toggled via env (unset=baseline/original gate, 1=fix)
#   * --max-model-len 16384 (enough for the 6K bench; faster startup; does NOT affect the
#     decode attention shape or the 3D threshold)
#   * name vllm-qwen-test, NO --rm (persist on crash for `docker logs`)
set -euo pipefail

IMG=tcclaviger/vllm-rocm-mxfp4-nvfp4:latest
PATCHED="${PATCHED_TRITON:-gemma-nvfp4/port-0191/qwen_triton_unified_attention.3dfix.py}"
TARGET=/app/vllm/vllm/v1/attention/ops/triton_unified_attention.py
[ -f "$PATCHED" ] || { echo "patched triton not found: $PATCHED" >&2; exit 1; }

R1=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render); C1=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-card)
R2=$(readlink -f /dev/dri/by-path/pci-0000:06:00.0-render); C2=$(readlink -f /dev/dri/by-path/pci-0000:06:00.0-card)
for d in "$R1" "$C1" "$R2" "$C2" /dev/kfd; do [ -e "$d" ] || { echo "missing $d" >&2; exit 1; }; done

THREEDENV=()
[ -n "${VLLM_TRITON_3D_SMALL_QLEN:-}" ] && THREEDENV=(-e "VLLM_TRITON_3D_SMALL_QLEN=${VLLM_TRITON_3D_SMALL_QLEN}")

# ENFORCE_EAGER=1 (default) eager; =0 compiled+cudagraph (match prod). MAXLEN default 16384.
EAGER=()
[ "${ENFORCE_EAGER:-1}" = "1" ] && EAGER=(--enforce-eager)

docker rm -f vllm-qwen-test >/dev/null 2>&1 || true
exec docker run -d --name vllm-qwen-test --network=host \
  --device=/dev/kfd --device="$R1" --device="$C1" --device="$R2" --device="$C2" \
  --group-add=video --group-add=render --ipc=host --security-opt seccomp=unconfined \
  -e HF_HUB_OFFLINE=1 \
  "${THREEDENV[@]}" \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v "$PATCHED":"$TARGET":ro \
  "$IMG" \
  Capicua25x/Qwen3.6-35B-A3B-DSV4Pro-Thinking-Distill-MXFP4-Vision \
  --served-model-name qwen --port 8011 --trust-remote-code \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len "${MAXLEN:-16384}" \
  --enable-prefix-caching --max-num-seqs 64 \
  "${EAGER[@]}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
