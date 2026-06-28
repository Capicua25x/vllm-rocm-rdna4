#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Gemma-4 26B-A4B NVFP4 MoE + gemma4_assistant MTP SPEC-DECODE draft on RDNA4
# vLLM 0.19.1 (capicua25x/vllm-rocm-rdna4:0.19.1), 2× R9700 (gfx1201, TP2).
#
# Spec-decode variant of the CONFIRMED-WORKING launch-26b-019-triton.sh. Deltas:
#   * Mounts the gemma4_assistant MTP backport (2 creates + 5 patched framework files)
#     ALONGSIDE the 4 RDNA NVFP4 graft files — ALL sourced from the single patched
#     0.19.1 working tree $P, bind-mounted over the image's /build editable install.
#   * --speculative-config (method=mtp, num_speculative_tokens=1 → single-step early-exit path).
#   * --max-model-len 16384 → 12288 (headroom for draft + per-group draft KV at TP2/0.92).
#   * --enforce-eager kept for bring-up (drop once it loads+generates to test compiled+cudagraph).
#   * NO --language-model-only (MTP proposer path does not run the multimodal vision probe).
#
# IMPORTANT (image is STOCK for the NVFP4 path — verified 2026-06-25): NONE of the RDNA
# grafts are baked in; all 4 MUST be mounted. The patched copies live in $P:
# oracle RDNA_TRITON, emul RDNA dequant, experts kernel, and the RDNA linear scheme
# (the patched 175-line invoke_nvfp4_linear/on_gfx1x scheme was installed into $P
# 2026-06-25, replacing the stock 124-line copy; .bak in port-0191/_backup_worktree_*).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# P = the patched 0.19.1 working tree (RDNA grafts + MTP backport all applied):
P="${VLLM_PATCHED_TREE:-<vllm-src>/vllm}"
[ -d "$P" ] || { echo "patched tree not found: $P" >&2; exit 1; }

# All files to overlay onto /build/vllm/vllm/<same path>. 4 RDNA graft + 7 MTP backport
# + 1 spec-decode 6K fix (triton 3D split-KV for query_len>1 verify, env-gated). = 12 files.
FILES=(
  # --- RDNA NVFP4 graft (4) ---
  model_executor/layers/fused_moe/oracle/nvfp4.py
  model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
  model_executor/layers/fused_moe/experts/rdna_nvfp4_moe.py
  model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py
  # --- gemma4_assistant MTP backport (2 creates + 5 patched) ---
  model_executor/models/gemma4_mtp.py
  v1/spec_decode/gemma4.py
  v1/spec_decode/eagle.py
  v1/worker/gpu_model_runner.py
  config/speculative.py
  model_executor/models/registry.py
  transformers_utils/model_arch_config_convertor.py
  # --- 6K spec-decode fix: 3D split-KV for query_len>1 verify (env-gated) ---
  v1/attention/ops/triton_unified_attention.py
)
MOUNTS=()
for f in "${FILES[@]}"; do
  [ -f "$P/$f" ] || { echo "missing patched file: $P/$f" >&2; exit 1; }
  MOUNTS+=( -v "$P/$f":"/build/vllm/vllm/$f":ro )
done

R1=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render); C1=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-card)
R2=$(readlink -f /dev/dri/by-path/pci-0000:06:00.0-render); C2=$(readlink -f /dev/dri/by-path/pci-0000:06:00.0-card)
for dev in "$R1" "$C1" "$R2" "$C2" /dev/kfd; do [ -e "$dev" ] || { echo "missing $dev" >&2; exit 1; }; done

# --enforce-eager toggle: ENFORCE_EAGER=1 (default) for bring-up; =0 for compiled+cudagraph.
EAGER=()
[ "${ENFORCE_EAGER:-1}" = "1" ] && EAGER=(--enforce-eager)

# SPEC toggle: SPEC=1 (default) enables the MTP draft; SPEC=0 = matched no-spec baseline
# (same model/flags, no --speculative-config) for an apples-to-apples regression comparison.
# SPEC_TOKENS: draft depth (1 = MTP-1, default; 3 = MTP-3). Requires the
# constant_draft_positions guards in eagle.py (HUNK A/B/C) for >1.
SPECARG=()
[ "${SPEC:-1}" = "1" ] && SPECARG=(--speculative-config "{\"model\":\"google/gemma-4-26B-A4B-it-assistant\",\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-1}}")

# 6K draft-attention experiment: GEMMA4_MTP_DRAFT_FULL_WINDOW=<int> caps the draft's
# single full-attention layer at a sliding window (cuts its long-context KV read).
# Forwarded into the container; unset = original full-attention draft behaviour.
WINENV=()
[ -n "${GEMMA4_MTP_DRAFT_FULL_WINDOW:-}" ] && WINENV=(-e "GEMMA4_MTP_DRAFT_FULL_WINDOW=${GEMMA4_MTP_DRAFT_FULL_WINDOW}")

# 6K spec-decode fix: VLLM_TRITON_3D_SMALL_QLEN=1 lets the 3D split-KV (flash-decode)
# triton kernel run for the spec VERIFY/propose forwards (query_len>1) when the tokens
# fit the segm buffers, instead of falling to the low-occupancy 2D kernel at long context.
THREEDENV=()
[ -n "${VLLM_TRITON_3D_SMALL_QLEN:-}" ] && THREEDENV=(-e "VLLM_TRITON_3D_SMALL_QLEN=${VLLM_TRITON_3D_SMALL_QLEN}")

# Bring-up: NO --rm so a crashed container persists for `docker logs`. Clean up any prior run first.
docker rm -f vllm-gemma >/dev/null 2>&1 || true
exec docker run -d --name vllm-gemma --network=host \
  --device=/dev/kfd --device="$R1" --device="$C1" --device="$R2" --device="$C2" \
  --group-add=video --group-add=render --ipc=host \
  --security-opt=no-new-privileges --cap-drop=ALL --cap-add=DAC_READ_SEARCH --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e HF_HUB_OFFLINE=1 \
  "${WINENV[@]}" \
  "${THREEDENV[@]}" \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  "${MOUNTS[@]}" \
  capicua25x/vllm-rocm-rdna4:0.19.1 \
  --model RedHatAI/gemma-4-26B-A4B-it-NVFP4 \
  --served-model-name gemma --port 8011 --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 12288 \
  --enable-prefix-caching --max-num-seqs "${MAXSEQS:-16}" \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  "${EAGER[@]}" \
  "${SPECARG[@]}"
  # Tool-calling: Gemma-4 ships a native tool-call format (<|tool_call>call:fn{k:<|"|>v<|"|>}<tool_call|>)
  # and vLLM 0.19.1 ships the matching 'gemma4' tool-call + reasoning parsers (lazy-registered in
  # vllm/tool_parsers/__init__.py and vllm/reasoning/__init__.py). Verified round-trip 2026-06-25
  # (toolcall_test.py): correct fn name + JSON-decodable args, finish_reason=tool_calls, fed-back
  # result used in NL answer, no spurious call on plain chat — both enable_thinking on and off.
  # OOM at load? drop --max-model-len 12288 → 8192, or --max-num-seqs 16 → 8.
  # Once it loads + accepts spec tokens clean, drop --enforce-eager for compiled+cudagraph.

# After it's up:
#   docker logs -f vllm-gemma   # → "Using 'RDNA_TRITON' NvFp4 MoE backend" + per-draft-layer KV-share map + startup complete
#   curl -s localhost:8011/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"gemma","messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":20}' | jq -r '.choices[0].message.content'
#   curl -s localhost:8011/metrics | grep -iE "spec_decode|accept|num_drafts"
#   docker stop vllm-gemma
