#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Gemma-4 26B-A4B NVFP4 MoE on RDNA4 — vLLM 0.19.1 (capicua25x/vllm-rocm-rdna4:0.19.1)
# ✅ CONFIRMED WORKING 2026-06-25 on 2× R9700 (gfx1201, TP2). First Gemma-4 NVFP4 MoE on RDNA4.
#
# This is the dev-rig form: bind-mount the 4 RDNA NVFP4 graft files over the image's STOCK
# vLLM. For prod, bake them into the image instead (see ../README.md "Next").
#
# THE FOUR THINGS THE STAGED VERSION GOT WRONG (all fixed below):
#   1. Mount target is /build/vllm/... NOT /app/vllm/... — the image runs the editable install at
#      /build; /app is a stale Apr-3 tree. The 0.19.1 bake applied MXFP4 but NOT the NVFP4 graft.
#   2. The model is passed as a --model FLAG, not a positional. ENTRYPOINT = api_server module;
#      with HF_HUB_OFFLINE=1 a positional model_tag sets a bad revision → LocalEntryNotFoundError.
#   3. FOUR graft files (added the linear scheme compressed_tensors_w4a4_nvfp4.py — without it,
#      linear NVFP4 → Marlin → gptq_marlin_repack missing on ROCm; emulation assumes swizzled scales).
#   4. NO --speculative-config in this baseline wrapper (spec-decode = port-0191/launch-26b-019-spec.sh,
#      via the gemma4_assistant draft backported from vLLM 0.23.0); and NO VLLM_USE_NVFP4_CT_EMULATIONS
#      (the linear scheme handles RDNA dequant directly).
#
# --enforce-eager is OPTIONAL (both eager and torch.compile+cudagraph are confirmed working). Drop it
# for the production (compiled) config; keep it for fastest cold-start during iteration.
# Vision works in this config; if you instead want to chase draft-spec, add --language-model-only
# (clears the multimodal gate) — but the draft still won't load until the native class exists.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
# G = dir holding the 4 patched graft files (this recipe dir, or your working copy):
G="${GEMMA_GRAFT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
BV=/build/vllm/vllm/model_executor/layers   # editable install path INSIDE the image

R1=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render); C1=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-card)
R2=$(readlink -f /dev/dri/by-path/pci-0000:06:00.0-render); C2=$(readlink -f /dev/dri/by-path/pci-0000:06:00.0-card)
for dev in "$R1" "$C1" "$R2" "$C2" /dev/kfd; do [ -e "$dev" ] || { echo "missing $dev" >&2; exit 1; }; done

# NOTE: stop any other vLLM instance bound to :8011 / the GPUs before launching.

exec docker run --rm -d --name vllm-gemma --network=host \
  --device=/dev/kfd --device="$R1" --device="$C1" --device="$R2" --device="$C2" \
  --group-add=video --group-add=render --ipc=host \
  --security-opt=no-new-privileges --cap-drop=ALL --cap-add=DAC_READ_SEARCH --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e HF_HUB_OFFLINE=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v "$G/oracle_nvfp4.py":"$BV/fused_moe/oracle/nvfp4.py":ro \
  -v "$G/nvfp4_emulation_utils.py":"$BV/quantization/utils/nvfp4_emulation_utils.py":ro \
  -v "$G/rdna_nvfp4_moe.py":"$BV/fused_moe/experts/rdna_nvfp4_moe.py":ro \
  -v "$G/compressed_tensors_w4a4_nvfp4.py":"$BV/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py":ro \
  capicua25x/vllm-rocm-rdna4:0.19.1 \
  --model RedHatAI/gemma-4-26B-A4B-it-NVFP4 \
  --served-model-name gemma --port 8011 --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 16384 \
  --enable-prefix-caching --max-num-seqs 16
  # (add --enforce-eager for fastest cold-start; omit for compiled+cudagraph prod config)

# NOTE: this script references $G/oracle_nvfp4.py and $G/nvfp4_emulation_utils.py — the PATCHED
# copies. This recipe dir tracks the diff in vllm-0.19.1-rdna4-nvfp4.patch; materialize the patched
# files (apply the patch to a /build stock checkout) or point GEMMA_GRAFT_DIR at your working tree.
#
# After it's up:
#   docker logs -f vllm-gemma            # watch → "Using 'RDNA_TRITON' NvFp4 MoE backend" + startup complete
#   curl -s localhost:8011/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"gemma","messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":20}' | jq -r '.choices[0].message.content'
#   docker stop vllm-gemma
