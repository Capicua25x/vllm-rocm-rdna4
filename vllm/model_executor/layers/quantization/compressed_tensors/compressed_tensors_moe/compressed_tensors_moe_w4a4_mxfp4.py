# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Modified 2026 by Capicua25x for the RDNA4 (gfx1200/gfx1201) port: gate MXFP4 MoE onto the RDNA
# triton_kernels graft path when running on gfx1200/gfx1201.


from functools import cache

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassExpertsMxfp4,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    MarlinExperts,
)
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
    XPUExpertsMxFp4,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


@cache
def _on_rdna4_with_triton_graft() -> bool:
    """gfx1200 / gfx1201 con el injerto RDNA de triton_kernels COMPLETO.

    Deliberadamente `on_rdna4()` y no el `on_gfx1x()` del parche 0.19.1:
    `on_gfx1x()` también admite gfx11 (RDNA3), donde la rama RDNA4 de
    `opt_flags` del injerto nunca dispara y este camino no se ha probado.
    Mismo criterio que `kernels/linear/mxfp4/rdna.py`.

    Se comprueban los TRES símbolos del injerto, no sólo el layout: con
    `RDNAMXValueLayout` presente pero `get_rdna_version_host` ausente,
    `opt_flags` elegiría tiles CDNA/por-defecto (split_k, persistent) que la
    rama RDNA_VALUE de `_matmul_ogs` nunca ha visto. Injerto parcial =
    injerto ausente.
    """
    if not current_platform.is_rocm():
        return False

    from vllm.platforms.rocm import on_rdna4

    if not on_rdna4():
        return False

    # OJO al orden: has_triton_kernels() *importa* el paquete (aliasea
    # vllm.third_party.triton_kernels en sys.modules), así que tiene que
    # correr ANTES de cualquier `from triton_kernels...`.
    from vllm.utils.import_utils import has_triton_kernels

    if not has_triton_kernels():
        return False

    try:
        from triton_kernels.target_info import (  # noqa: F401
            get_rdna_version_host,
        )
        from triton_kernels.tensor_details.layout import (  # noqa: F401
            RDNAMXValueLayout,
        )
        from triton_kernels.tensor_details.layout_details.rdna_value import (  # noqa: F401,E501
            mxfp4_dequant_rdna,
        )
    except ImportError:
        return False

    return True

    # TODO(rdna4-mxfp4-moe): esta comprobación duplica
    # `_has_rdna_mxfp4_graft()` de
    # `vllm/model_executor/kernels/linear/mxfp4/rdna.py`. NO se importa de allí
    # a propósito: ese módulo llama a `direct_register_custom_op` en el import
    # (registra torch.ops.vllm.rdna_mxfp4_gemm), y la ruta MoE no debe arrastrar
    # ese efecto global. Si el fichero denso se fusiona, mover el helper a un
    # sitio compartido y borrar esta copia.


class CompressedTensorsW4A4Mxfp4MoEMethod(CompressedTensorsMoEMethod):
    def __init__(self, moe):
        super().__init__(moe)
        self.group_size = 32
        self.mxfp4_backend = Mxfp4MoeBackend.MARLIN
        # use cutlass if supported, otherwise fallback to marlin for weight-only FP4
        self.use_cutlass_mxfp4 = CutlassExpertsMxfp4._supports_current_device()
        # RDNA4 (gfx12xx) has no Marlin kernel: `gptq_marlin_repack` is a
        # CUDA-only op and `torch.ops._C` on ROCm does not carry it, so
        # `prepare_moe_fp4_layer_for_marlin` aborts at load time. Route to the
        # Triton-unfused experts, which consume the checkpoint layout through
        # `_swizzle_mxfp4` + `matmul_ogs` and never touch Marlin.
        self.use_rdna4_triton = (
            not self.use_cutlass_mxfp4 and _on_rdna4_with_triton_graft()
        )
        # TRITON_* backends free w13/w2_weight_scale after swizzling; the
        # swizzled scales live inside these PrecisionConfigs instead.
        # Mirrors Mxfp4MoEMethod (layers/quantization/mxfp4.py).
        self.w13_precision_config = None
        self.w2_precision_config = None
        self.experts_cls: type[mk.FusedMoEExperts]
        if self.use_cutlass_mxfp4:
            logger.info_once("Using CutlassExpertsMxfp4 for MXFP4 MoE")
            self.experts_cls = CutlassExpertsMxfp4
        elif current_platform.is_xpu():
            self.mxfp4_backend = Mxfp4MoeBackend.XPU
            self.experts_cls = XPUExpertsMxFp4
            logger.info_once("Using XPUExpertsMxFp4 for MXFP4 MoE on XPU platform")
        elif self.use_rdna4_triton:
            from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
                UnfusedOAITritonExperts,
            )

            # Esta clase fija experts_cls A MANO, sin pasar por
            # `is_supported_config` del oráculo (igual que ya hace con Marlin).
            # El assert es la red: si upstream estrecha el gate de dispositivo,
            # esto falla RUIDOSAMENTE en vez de correr una clase que se
            # considera a sí misma no soportada.
            assert UnfusedOAITritonExperts._supports_current_device(), (
                "UnfusedOAITritonExperts no admite este dispositivo pese a "
                "on_rdna4() + injerto RDNA presente"
            )

            self.mxfp4_backend = Mxfp4MoeBackend.TRITON_UNFUSED
            self.experts_cls = UnfusedOAITritonExperts
            logger.info_once(
                "Using UnfusedOAITritonExperts for MXFP4 MoE on RDNA4 "
                "(no Marlin kernel on gfx12xx)"
            )
            # TODO(rdna4-mxfp4-moe): el parche 0.19.1 del operador EVITABA
            # `UnfusedOAITritonExperts` porque su gather torch crudo
            # (`intermediate_cache1.view(-1, N)[gather_indx.dst_indx]`, hoy en
            # gpt_oss_triton_kernels_moe.py:1217) se salía de rango con routing
            # ragged, y prefería `OAITritonMxfp4ExpertsMonolithic` con la
            # activación desfusionada a mano. En 0.26.1 esa clase monolítica
            # exige SWIGLUOAI (gpt_oss_triton_kernels_moe.py:1326) y
            # `triton_kernel_fused_experts` sigue con
            # `assert activation == MoEActivation.SWIGLUOAI` (línea 650);
            # Ornith es SILU, así que esa ruta no se puede mapear sin rehacer
            # el hunk #1 entero. La ruta unfused SÍ se reescribió aguas arriba
            # (remap_topk_to_local, masked_moe_sum, centinela -1), pero NO se ha
            # verificado que el OOB de 0.19.1 esté cerrado. Verificar con topk
            # ragged / EP antes de promocionar a producción.
        else:
            logger.info_once("Using MarlinExperts for MXFP4 MoE")
            self.experts_cls = MarlinExperts

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.use_cutlass_mxfp4:
            # W4A4: both weights and activations quantized to MXFP4
            return mxfp4_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
        elif self.use_rdna4_triton:
            # TRITON_UNFUSED: `convert_weight_to_mxfp4_moe_kernel_format` ya
            # borró layer.w13/w2_weight_scale; las escalas swizzleadas viven
            # dentro de los PrecisionConfig. Idéntico a Mxfp4MoEMethod.
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            return make_mxfp4_moe_quant_config(
                mxfp4_backend=self.mxfp4_backend,
                w1_scale=self.w13_precision_config,
                w2_scale=self.w2_precision_config,
                layer=layer,
            )
        else:
            # W4A16: weight-only via Marlin
            return make_mxfp4_moe_quant_config(
                mxfp4_backend=self.mxfp4_backend,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                layer=layer,
            )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        if self.use_cutlass_mxfp4:
            # Swizzle weight scales from flat checkpoint layout [E, N, K//32]
            # to CUTLASS tiled layout [E, numMTiles*numKTiles*512].
            from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
                swizzle_mxfp4_scales,
            )

            E = layer.w13_weight_scale.shape[0]
            w13_N = layer.w13_weight_scale.shape[1]
            w13_scale_K = layer.w13_weight_scale.shape[2]
            w13_K = w13_scale_K * 32

            w2_M = layer.w2_weight_scale.shape[1]
            w2_scale_N = layer.w2_weight_scale.shape[2]
            w2_N = w2_scale_N * 32

            swizzled_w13 = []
            swizzled_w2 = []
            for e_idx in range(E):
                s13 = layer.w13_weight_scale[e_idx]
                sw13 = swizzle_mxfp4_scales(s13, w13_N, w13_K)
                swizzled_w13.append(sw13.reshape(w13_N, w13_scale_K))
                s2 = layer.w2_weight_scale[e_idx]
                sw2 = swizzle_mxfp4_scales(s2, w2_M, w2_N)
                swizzled_w2.append(sw2.reshape(w2_M, w2_scale_N))
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.stack(swizzled_w13), requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                torch.stack(swizzled_w2), requires_grad=False
            )
        elif current_platform.is_xpu():
            pass
        elif self.use_rdna4_triton:
            # RDNA4: NO `prepare_moe_fp4_layer_for_marlin` — su
            # `_repack_marlin_experts` llama a `ops.gptq_marlin_repack`, que en
            # ROCm no existe en `torch.ops._C`. En su lugar, la misma conversión
            # que usa Mxfp4MoEMethod para los backends TRITON_*: `_swizzle_mxfp4`
            # envuelve pesos y escalas en tensores de triton_kernels y devuelve
            # PrecisionConfigs. Los pesos siguen empaquetados a 4 bits; el
            # dequant ocurre dentro del GEMM (`mxfp4_dequant_rdna`).
            #
            # `convert_weight_to_mxfp4_moe_kernel_format` hace `del` de
            # layer.w13/w2_weight y de layer.w13/w2_weight_scale, así que hay que
            # volver a asignar los pesos como atributos planos (los tensores de
            # triton_kernels no son nn.Parameter y no soportan .detach()).
            #
            # Este método no crea sesgos MoE, así que w13_bias/w2_bias son None y
            # la rama TRITON_UNFUSED de la conversión los deja pasar intactos.
            w13, w2, w13_precision, w2_precision, _, _ = (
                convert_weight_to_mxfp4_moe_kernel_format(
                    mxfp4_backend=self.mxfp4_backend,
                    layer=layer,
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    w13_weight_scale=layer.w13_weight_scale,
                    w2_weight_scale=layer.w2_weight_scale,
                    w13_bias=None,
                    w2_bias=None,
                )
            )
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_precision
            self.w2_precision_config = w2_precision
        else:
            logger.warning_once(
                "Your GPU does not have native support for FP4 computation "
                "but FP4 quantization is being used. Weight-only FP4 "
                "compression will be used leveraging the Marlin kernel. "
                "This may degrade performance for compute-heavy workloads."
            )
            prepare_moe_fp4_layer_for_marlin(layer)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                mxfp4_backend=self.mxfp4_backend,
                routing_tables=layer._expert_routing_tables(),
            )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
