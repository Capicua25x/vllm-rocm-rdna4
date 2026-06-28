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
# (the patched 175-line invoke_nvfp4_linear/on_gfx1x scheme was installed into the tree
# 2026-06-25, replacing the stock 124-line copy; .bak in port-0191/_backup_worktree_*).
#
# Free the GPUs + :8011 first: stop any other vLLM service bound to them.
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
  # --- cold-prefill fix: grow segm buffers for large-q 3D prefill (VLLM_TRITON_3D_PREFILL, env-gated) ---
  v1/attention/backends/triton_attn.py
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

# CUDAGRAPH_MODE: override the cudagraph capture mode (default when compiled = FULL_AND_PIECEWISE,
# which uses PIECEWISE graphs for prefill). FULL_DECODE_ONLY = full graph for decode (keeps the
# fast ~9.6 tok/s long-ctx decode) + EAGER prefill (sidesteps the compiled cold-prefill hang that
# kills compaction). Valid: FULL | FULL_AND_PIECEWISE | FULL_DECODE_ONLY | NONE | PIECEWISE.
CGMODE=()
[ -n "${CUDAGRAPH_MODE:-}" ] && CGMODE=(--compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\"}")

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

# Cold-prefill fix: VLLM_TRITON_3D_PREFILL=1 grows the FullAttentionSpec segm buffers to
# max_num_batched_tokens so large-query COLD-PREFILL chunks ride the 3D split-KV kernel
# instead of the low-occupancy serial 2D kernel (the 93%-of-cold-prefill bottleneck).
# Self-sufficient: opens the dispatch gate AND grows the buffers (does NOT also need
# VLLM_TRITON_3D_SMALL_QLEN). Requires the patched triton_attn.py mount in FILES above.
# Default unset = no behavior change. mm_prefix (image) prefill is force-routed to 2D.
PREFILL3DENV=()
[ -n "${VLLM_TRITON_3D_PREFILL:-}" ] && PREFILL3DENV=(-e "VLLM_TRITON_3D_PREFILL=${VLLM_TRITON_3D_PREFILL}")

# Cold-prefill fix experiment (env-gated in the patched triton_unified_attention.py):
# VLLM_PREFILL_TILE_SIZE=128 (or 64/256) overrides ONLY the prefill KV tile to test
# whether fatter tiles flatten the super-quadratic depth curve. Decode untouched. Unset = 32 (stock).
PREFILLTILE=()
[ -n "${VLLM_PREFILL_TILE_SIZE:-}" ] && PREFILLTILE=(-e "VLLM_PREFILL_TILE_SIZE=${VLLM_PREFILL_TILE_SIZE}")

# Lever A (head-512 prefill kernel fix): VLLM_HEAD512_CHUNK=N loads Q/K in N-wide head
# slices inside the QK dot so the head-512 (Gemma full-attn) layers stop staging a 128KiB
# Q operand in LDS (RDNA4 64KiB cap), letting head-512 prefill use TILE=64 (was pinned 32).
# N=128 verified (probe) to drop shared 131072→32768 B. Unset = monolithic (stock, slow).
HEAD512CHUNK=()
[ -n "${VLLM_HEAD512_CHUNK:-}" ] && HEAD512CHUNK=(-e "VLLM_HEAD512_CHUNK=${VLLM_HEAD512_CHUNK}")

# Head-512 flash-prefill rewrite: VLLM_FA_HEADCHUNK=64 routes head-512 (Gemma full-attn) prefill to the
# dedicated chunked kernel (chunks BOTH QK and PV in 64-wide head slices -> full 512-deep K/V never staged
# -> low LDS -> high occupancy on RDNA4). 0/unset = stock monolithic. Prototype: bit-identical, LDS -91%.
FAHEADCHUNK=()
[ -n "${VLLM_FA_HEADCHUNK:-}" ] && FAHEADCHUNK=(-e "VLLM_FA_HEADCHUNK=${VLLM_FA_HEADCHUNK}")

# Chunk-size lever (cold-prefill Step D): MAX_NUM_BATCHED overrides the effective 2048
# default. R9700's 32GB < the 70GiB OPENAI_API_SERVER threshold pins max_num_batched_tokens
# to 2048 (arg_utils.py default table); every prefill chunk is then 2048 tokens, all routed
# to the slow 2D triton attention kernel. Bigger chunks amortize MoE/launch overhead but 4x
# the staging+activation VRAM (verify no OOM at TP2/0.92) AND enlarge the 2D-kernel query
# batch — A/B it. Unset = stock 2048.
MNBT=()
[ -n "${MAX_NUM_BATCHED:-}" ] && MNBT=(--max-num-batched-tokens "${MAX_NUM_BATCHED}")

# KV-cache dtype: KV_CACHE_DTYPE=fp8 halves KV-cache bytes → ~2× cache tokens (160k→~320k),
# the capacity lever for >175k context. Unset = stock auto (model dtype). fp8 = fp8_e4m3 on ROCm.
KVDTYPE=()
[ -n "${KV_CACHE_DTYPE:-}" ] && KVDTYPE=(--kv-cache-dtype "${KV_CACHE_DTYPE}")

# Torch profiler (cold-prefill Step B): PROFILER=1 arms the POST /start_profile + /stop_profile
# endpoints. NOTE this vLLM 0.19.1 uses --profiler-config (NOT the legacy VLLM_TORCH_PROFILER_DIR
# env). with_flops=true makes the trace report per-op FLOPS, which directly settles the
# attention-2D-kernel vs NVFP4-MoE question. Traces (CPU+GPU) land in $PROF_HOST_DIR on the host.
# Usage: PROFILER=1 ./launch...; then  curl -XPOST :8011/start_profile; <fire ONE cold request>;
#        curl -XPOST :8011/stop_profile; inspect the trace under $PROF_HOST_DIR.
PROF_HOST_DIR="${PROF_HOST_DIR:-$(cd "$(dirname "$0")" && pwd)/prof}"
PROFARG=()
PROFMOUNT=()
if [ "${PROFILER:-0}" = "1" ]; then
  mkdir -p "$PROF_HOST_DIR"
  PROFMOUNT=(-v "$PROF_HOST_DIR":/profiles)
  PROFARG=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/profiles","torch_profiler_with_flops":true,"torch_profiler_record_shapes":true,"torch_profiler_with_stack":false,"ignore_frontend":true,"max_iterations":24}')
fi

# Bring-up: NO --rm so a crashed container persists for `docker logs`. Clean up any prior run first.
docker rm -f vllm-gemma >/dev/null 2>&1 || true
exec docker run -d --name vllm-gemma --network=host \
  --device=/dev/kfd --device="$R1" --device="$C1" --device="$R2" --device="$C2" \
  --group-add=video --group-add=render --ipc=host \
  --security-opt=no-new-privileges --cap-drop=ALL --cap-add=DAC_READ_SEARCH --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e HF_HUB_OFFLINE=1 \
  "${WINENV[@]}" \
  "${THREEDENV[@]}" \
  "${PREFILL3DENV[@]}" \
  "${PREFILLTILE[@]}" \
  "${HEAD512CHUNK[@]}" \
  "${FAHEADCHUNK[@]}" \
  "${PROFMOUNT[@]}" \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  "${MOUNTS[@]}" \
  capicua25x/vllm-rocm-rdna4:0.19.1 \
  --model RedHatAI/gemma-4-26B-A4B-it-NVFP4 \
  --served-model-name ${SERVED_NAMES:-gemma qwen} --port 8011 --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --tensor-parallel-size 2 --gpu-memory-utilization "${GPU_MEM:-0.92}" --max-model-len "${MODEL_LEN:-262144}" \
  --enable-prefix-caching --max-num-seqs "${MAXSEQS:-16}" \
  "${KVDTYPE[@]}" \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  "${EAGER[@]}" \
  "${CGMODE[@]}" \
  "${MNBT[@]}" \
  "${PROFARG[@]}" \
  "${SPECARG[@]}"
  # OOM at load? drop --max-model-len 262144 → 8192, or --max-num-seqs 16 → 8.
  # Once it loads + accepts spec tokens clean, drop --enforce-eager for compiled+cudagraph.

# After it's up:
#   docker logs -f vllm-gemma   # → "Using 'RDNA_TRITON' NvFp4 MoE backend" + per-draft-layer KV-share map + startup complete
#   curl -s localhost:8011/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"gemma","messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":20}' | jq -r '.choices[0].message.content'
#   curl -s localhost:8011/metrics | grep -iE "spec_decode|accept|num_drafts"
#   docker stop vllm-gemma
