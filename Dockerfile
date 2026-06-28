# capicua25x/vllm-rocm-rdna4:0.19.1  —  baked image for the RDNA4 MXFP4 Qwen distill.
# Freezes the validated build steps (editable vLLM install + RDNA4 graft) into image
# layers. Base = Rob's working RDNA4 vLLM image.
# Order is load-bearing: librccl symlink + cmake/ninja reachable BEFORE the editable install
# (cmake-configure's roc::rccl imported target needs librccl.so.1.0.70201). --no-deps keeps Rob's
# torch 2.10 / ROCm 7.2.1. Do NOT set HIPBLASLT_TENSILE_LIBPATH (Rob's hipBLASLt 1.2 is the default).
FROM tcclaviger/vllm-rocm-mxfp4-nvfp4:latest

ENV PATH=/app/.venv/bin:$PATH \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.19.1 \
    VLLM_MXFP4_USE_MARLIN=0 \
    HF_HUB_OFFLINE=1

COPY vllm-0.19.1 /build/vllm
COPY overlay_graft.py /work/overlay_graft.py
COPY graft/tk /graft/tk

# 1. transformers -> 5.8.1
RUN python -m pip install --no-input "transformers==5.8.1" \
 && python -c "import transformers; assert transformers.__version__.startswith('5.8.1'), transformers.__version__; print('transformers', transformers.__version__)"

# 2. editable vLLM 0.19.1 — mirrors the proven build steps. &&-chained so ANY failure aborts the
#    build (no pipe/tail masking). Inline export PATH matches the proven build script. The C-ext
#    recompiles from csrc against Rob's torch/ROCm (GPU-free, ~3 min) — deterministic, reproducible.
RUN export PATH=/app/.venv/bin:$PATH \
 && export VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1201 CMAKE_BUILD_TYPE=Release MAX_JOBS="$(nproc)" \
 && git config --global --add safe.directory '*' \
 && ln -sf "$(ls /opt/rocm-7.2.1/lib/librccl.so.1.0.* | head -1)" /opt/rocm-7.2.1/lib/librccl.so.1.0.70201 \
 && ls -l /opt/rocm-7.2.1/lib/librccl.so.1.0.70201 \
 && ln -sf /app/.venv/bin/cmake  /usr/local/bin/cmake \
 && ln -sf /app/.venv/bin/ninja  /usr/local/bin/ninja \
 && ln -sf /app/.venv/bin/ccmake /usr/local/bin/ccmake \
 && python -m pip install -e /build/vllm --no-deps --no-build-isolation \
 && python -c "import vllm; print('VLLM_VERSION', vllm.__version__); assert vllm.__version__.startswith('0.19.1'), vllm.__version__"

# 3. RDNA4 triton_kernels graft — AFTER the editable install (install re-vendors tk from pristine .deps).
#    triton_kernels is only importable via vLLM's vendored sys.path — verify by markers, not import.
RUN TK=/build/vllm/vllm/third_party/triton_kernels \
 && cp /graft/tk/tensor_details/layout.py            "$TK/tensor_details/layout.py" \
 && cp /graft/tk/tensor_details/layout_details/rdna_value.py "$TK/tensor_details/layout_details/rdna_value.py" \
 && python /work/overlay_graft.py \
 && grep -q 'def get_rdna_version_host' "$TK/target_info.py" \
 && grep -q 'get_rdna_version_host() == 4' "$TK/matmul_ogs_details/opt_flags.py" \
 && grep -q 'SWIZZLE_MX_VALUE == "RDNA_VALUE"' "$TK/matmul_ogs_details/_matmul_ogs.py" \
 && grep -q 'mxfp4_dequant_rdna' "$TK/tensor_details/layout_details/rdna_value.py" \
 && grep -q 'RDNAMXValueLayout' "$TK/tensor_details/layout.py" \
 && echo "triton_kernels RDNA graft verified (5 markers)"

# 4. correctness gates (GPU-free): the 4 RDNA4 vLLM-layer source edits present + entrypoint resolves
RUN MOE=/build/vllm/vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py \
 && CT=/build/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py \
 && SW=/build/vllm/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py \
 && grep -q '_is_rdna4' "$MOE" \
 && grep -q 'on_gfx1x() and activation == MoEActivation.SILU' "$MOE" \
 && grep -q 'CompressedTensorsMxfp4RdnaMoeMethod' "$CT" \
 && grep -q 'on_gfx1x' "$SW" \
 && echo "RDNA4 vLLM-layer edits present" \
 && python -m vllm.entrypoints.openai.api_server --help >/dev/null 2>&1 \
 && echo "api_server entrypoint resolves OK"

ENTRYPOINT ["/app/.venv/bin/python", "-m", "vllm.entrypoints.openai.api_server"]
