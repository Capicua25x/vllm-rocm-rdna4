# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    invoke_nvfp4_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    apply_nvfp4_linear,
    convert_to_nvfp4_linear_kernel_format,
    select_nvfp4_linear_backend,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):
    def __init__(self):
        # RDNA3 / RDNA4 has no NVFP4 linear backend available (no FlashInfer,
        # no CUTLASS FP4, no Marlin FP4 repack op in ROCm builds). Short-circuit
        # before select_nvfp4_linear_backend() is called — otherwise it logs a
        # misleading "Using Marlin" warning and the loader later crashes on
        # `torch.ops._C.gptq_marlin_repack` which is not compiled into ROCm.
        from vllm.platforms import current_platform
        _rdna = False
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx1x
            _rdna = on_gfx1x()
        self.backend = None if _rdna else select_nvfp4_linear_backend()
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # RDNA3/RDNA4: no NVFP4-capable linear backend on ROCm RDNA (no
        # FlashInfer, no CUTLASS FP4, no Marlin repack). Use a Triton kernel
        # that keeps weights packed as uint8 and dequants at runtime.
        from vllm.platforms import current_platform
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx1x
            if on_gfx1x():
                self._rdna_runtime_dequant = True
                # Keep packed weights in VRAM; dequant at runtime in Triton.
                w_gscale_inv = (
                    1.0 / layer.weight_global_scale.max()
                ).to(torch.float32)
                layer.weight = Parameter(
                    layer.weight_packed.data.contiguous(),
                    requires_grad=False,
                )
                del layer.weight_packed
                layer.weight_scale = Parameter(
                    layer.weight_scale.data.contiguous(),
                    requires_grad=False,
                )
                layer.weight_global_scale_inv = Parameter(
                    w_gscale_inv, requires_grad=False
                )
                for name in (
                    "weight_global_scale",
                    "input_global_scale",
                ):
                    if hasattr(layer, name):
                        delattr(layer, name)
                return

        # Rename CT checkpoint names to standardized names
        layer.weight = layer.weight_packed
        del layer.weight_packed
        # Process global scales (CT stores as divisors, i.e. 1/scale)
        input_global_scale_inv = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(
            (1.0 / input_global_scale_inv).to(torch.float32), requires_grad=False
        )
        weight_global_scale = layer.weight_global_scale.max().to(torch.float32)
        layer.weight_global_scale = Parameter(
            1.0 / weight_global_scale, requires_grad=False
        )

        # Pre-compute alpha and inverse for runtime quantization
        layer.input_global_scale_inv = Parameter(
            input_global_scale_inv, requires_grad=False
        )
        layer.alpha = Parameter(
            layer.input_global_scale * layer.weight_global_scale, requires_grad=False
        )

        # Convert layer to NVFP4 linear kernel format
        convert_to_nvfp4_linear_kernel_format(self.backend, layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "_rdna_runtime_dequant", False):
            return invoke_nvfp4_linear_kernel(
                x, layer.weight, layer.weight_scale,
                layer.weight_global_scale_inv, bias,
            )

        return apply_nvfp4_linear(
            backend=self.backend,
            layer=layer,
            x=x,
            bias=bias,
        )
