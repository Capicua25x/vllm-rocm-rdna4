# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RDNA3 / RDNA4 monolithic NVFP4 fused-MoE experts.

Wraps the hand-built Triton kernel `invoke_fused_moe_nvfp4_kernel` so it can
be selected by the NVFP4 MoE oracle when running on gfx11xx / gfx12xx.
"""
import torch
import triton.language as tl

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    invoke_fused_moe_nvfp4_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

# Default Triton tuning config for the RDNA fused NVFP4 MoE kernel.
# BLOCK_SIZE_K must be a multiple of 16 (one NVFP4 group per step).
# Revisit after first benchmarks on gfx12.
_RDNA_NVFP4_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
}


class RdnaNvFp4ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """
    Monolithic NVFP4 fused-MoE experts for AMD RDNA3 / RDNA4.

    Calls the hand-built Triton kernel twice (w13 gate+up, then w2 down) with
    an activation-and-mul in between. No FlashInfer / CUTLASS / Marlin
    dependency — runs on any ROCm GPU exposing WMMA v1+ (gfx11, gfx12).
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        # Base __init__ assigns self.moe_config / self.quant_config and
        # validates the Standard vs BatchedExperts activation-format invariants.
        super().__init__(moe_config=moe_config, quant_config=quant_config)

        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts

        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None
        assert self.quant_config.g1_alphas is not None  # 1 / w13_global_scale, [E]
        assert self.quant_config.g2_alphas is not None  # 1 /  w2_global_scale, [E]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Nothing to do. Weight / scale conversion was already handled by the
        # RDNA_TRITON arm of convert_to_nvfp4_moe_kernel_format:
        #   - weights: raw packed uint8 FP4
        #   - per-group scales: fp8_e4m3fn, LINEAR layout (no swizzle)
        #   - g1/g2_alphas: fp32 [E]  (= 1 / w_global_scale, folded in-kernel)
        return

    # ---------------------------------------------------------------
    # Capability gates (consumed by select_nvfp4_moe_backend)
    # ---------------------------------------------------------------

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx1x
        return on_gfx1x()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        # Kernel itself is activation-agnostic; gate+up + mul is applied outside.
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # W4A4 NVFP4. Activations get quantized outside the kernel; static-weight
        # / dynamic-activation is the only combo this class is wired for.
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.GELU)

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        # NVFP4 group size = 16 along K; kernel's BLOCK_SIZE_K must align.
        return hidden_dim % 16 == 0

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # Monolithic: no all2all, no EPLB.
        return (
            not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def _supports_routing_method(
        routing_method_type: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Plain softmax/sigmoid top-k with optional renormalize, plus
        # `Unspecified` — which is what get_routing_method_type returns for
        # the "sigmoid + e_score_correction_bias + no grouped routing"
        # pattern used by MiniMax-M2 (and any other bias-using ungrouped
        # model). Grouped (DeepSeek-V3) and Llama4 routing remain out of
        # scope for this kernel.
        return routing_method_type in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.Unspecified,
        )

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        # fused_topk internally casts; fp32 / bf16 / fp16 all accepted.
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    @property
    def expects_unquantized_inputs(self) -> bool:
        # The RDNA Triton kernel consumes raw bf16 / fp16 activations directly.
        # No FP4 packing, no per-block activation scales, no a1q_scale tensor.
        # Returning True tells the upstream PrepareFinalize stage to pass
        # hidden_states through unchanged instead of quantizing them.
        return True

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        assert activation in (MoEActivation.SILU, MoEActivation.GELU)
        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None
        assert self.quant_config.g1_alphas is not None
        assert self.quant_config.g2_alphas is not None

        # RDNA path takes raw bf16 / fp16 activations. PrepareFinalize should
        # have honored expects_unquantized_inputs and left them unquantized.
        assert a1q_scale is None, (
            "RdnaNvFp4ExpertsMonolithic expects raw activations; "
            "got a1q_scale. PrepareFinalize ignored "
            "expects_unquantized_inputs=True."
        )

        # Grouped top-k / topk_group remain out of scope (DeepSeek-V3 style).
        assert num_expert_group is None
        assert topk_group is None
        # routed_scaling_factor IS supported — applied to topk_weights below.

        renormalize = True  # enforced by _supports_routing_method
        if e_score_correction_bias is not None:
            # sigmoid/softmax + bias + ungrouped top-k (MiniMax-M2 pattern).
            # moe_config.scoring_func tells us which score function to apply
            # before adding the bias; the canonical implementation lives in
            # fused_topk_bias_router.fused_topk_bias.
            from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (  # noqa: E501
                fused_topk_bias,
            )
            scoring_func = getattr(self.moe_config, "scoring_func", "softmax")
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=e_score_correction_bias,
                topk=self.topk,
                renormalize=renormalize,
                scoring_func=scoring_func,
            )
        else:
            topk_weights, topk_ids, _ = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.topk,
                renormalize=renormalize,
            )

        # Apply routed scaling if the model configured one (e.g. Qwen3.5,
        # DeepSeek-V3). No-op when None or 1.0.
        if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor

        M, K = hidden_states.shape
        E, two_I, _ = w1.shape
        N = two_I                   # gate+up projection output (= 2 * intermediate)
        top_k_num = self.topk

        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )

        # Intermediate buffers.
        intermediate_cache1 = torch.empty(
            (M, top_k_num, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache2 = torch.empty(
            (M * top_k_num, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache3 = torch.empty(
            (M, top_k_num, K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, _RDNA_NVFP4_DEFAULT_CONFIG["BLOCK_SIZE_M"],
            global_num_experts, expert_map,
        )

        # --- w13 projection (gate + up) ----------------------------------
        # w1_scale is fp8_e4m3fn per-group block scales (LINEAR layout).
        # g1_alphas is fp32 [E] = 1 / w13_global_scale (folded in-kernel).
        invoke_fused_moe_nvfp4_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            self.quant_config.w1_scale,
            self.quant_config.g1_alphas,
            topk_weights if apply_router_weight_on_input else None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            _RDNA_NVFP4_DEFAULT_CONFIG,
            compute_type,
        )

        # --- activation + mul --------------------------------------------
        if activation == MoEActivation.SILU:
            torch.ops._C.silu_and_mul(
                intermediate_cache2,
                intermediate_cache1.view(-1, N),
            )
        else:  # GELU
            torch.ops._C.gelu_and_mul(
                intermediate_cache2,
                intermediate_cache1.view(-1, N),
            )

        # --- w2 projection (down) ----------------------------------------
        # w2_scale is fp8_e4m3fn per-group block scales (LINEAR layout).
        # g2_alphas is fp32 [E] = 1 / w2_global_scale (folded in-kernel).
        invoke_fused_moe_nvfp4_kernel(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            self.quant_config.w2_scale,
            self.quant_config.g2_alphas,
            topk_weights if not apply_router_weight_on_input else None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            _RDNA_NVFP4_DEFAULT_CONFIG,
            compute_type,
        )

        # Reduce across top-k experts to produce [M, K] output.
        output = torch.empty_like(hidden_states)
        torch.ops._moe_C.moe_sum(intermediate_cache3, output)
        return output
