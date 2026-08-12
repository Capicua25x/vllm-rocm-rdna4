# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MXFP4 dense linear kernel for AMD RDNA4 (gfx1200 / gfx1201).

RDNA4 has no hardware microscaling datapath, but the RDNA graft carried by this
image's ``triton_kernels`` wheel provides an in-kernel Triton dequantizer
(``triton_kernels.tensor_details.layout_details.rdna_value.mxfp4_dequant_rdna``)
that is selected by tagging the weight tensor with ``RDNAMXValueLayout``.
``matmul_ogs`` then unpacks each weight tile to the activation dtype *inside*
the GEMM and feeds it to ``tl.dot`` on WMMA v2, so the 4-bit weights are never
materialized in high precision in VRAM — unlike
``EmulationMxfp4LinearKernel``, which dequantizes the whole weight to bf16 on
every forward.

This kernel is **weight-only (A16)**: activations are consumed at their native
bf16/fp16 precision.  Its numerical oracle is therefore

    F.linear(x, dequant_mxfp4(w, s, x.dtype), bias)

i.e. emulation *without* the activation QDQ that ``EmulationMxfp4LinearKernel``
applies when it is configured with ``kMxfp4Dynamic`` (see the note in
``can_implement``).

Wiring — three edits live outside this file, in
``vllm/model_executor/kernels/linear/__init__.py``:
  1. ``from .mxfp4.rdna import RdnaMxfp4LinearKernel``
  2. insert it into ``_POSSIBLE_MXFP4_KERNELS[PlatformEnum.ROCM]`` **before**
     ``EmulationMxfp4LinearKernel`` (and before ``AiterMxfp4LinearKernel`` is
     harmless — AITER declines on gfx12 today)
  3. add the name to ``__all__``

IMPORTANTE — acoplamiento con ``Platform.supports_mx()``:
    Este kernel NO consulta ``current_platform.supports_mx()`` y no debe
    hacerlo.  Hoy ``supports_mx()`` significa "hardware con microscaling
    NATIVO" (``gfx95`` / ``gfx1250``); gfx1201 no es eso — aquí el MXFP4 se
    resuelve con un dequant Triton dentro del kernel.  El gate de este archivo
    es ``on_rdna4()`` + presencia del graft, precisamente para no depender de
    ese predicado.

    Si alguna vez se extiende ``supports_mx()`` a gfx12, ese cambio y este
    archivo tienen que aterrizar JUNTOS, en el mismo commit: ensanchar
    ``supports_mx()`` por su cuenta haría que ``AiterMxfp4LinearKernel``
    reclamara gfx1201 (su ``is_supported`` empieza justo por ese predicado) y
    rompería el fallback de emulación que hoy SÍ funciona.  En ese mismo commit
    hay que auditar los demás consumidores, porque varios pasarían a reclamar
    MXFP8 nativo, que el graft no provee:
      - ``kernels/linear/mxfp4/aiter.py``
      - ``kernels/linear/mxfp4/emulation.py``
      - ``kernels/linear/mxfp8/rocm_native.py``
      - ``layers/quantization/quark/schemes/quark_ocp_mx.py``
      - ``layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py``
      - ``layers/fused_moe/experts/mxfp8_native_moe.py``
      - ``layers/fused_moe/experts/aiter_mxfp8_moe.py``
"""

from functools import cache

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_triton_kernels

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

logger = init_logger(__name__)

# OCP MX group size for MXFP4: 32 FP4 values share one E8M0 scale byte.
MXFP4_GROUP_SIZE = 32

# E2M1 lookup table, index = nibble value, low nibble first.  Same table as the
# operator's 0.19.1 RDNA4 patch and as quark's `dq_mxfp4` (which backs
# `torch.ops.vllm.dequant_mxfp4`).
# TODO(rdna-mxfp4): add a unit test asserting this table reproduces
# `dequant_mxfp4` bit-for-bit on a random uint8 weight — it is only used by the
# fallback path below, but a typo here would be silent.
_FP4_E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# Only bf16/fp16 activations can go through the graft: `mxfp4_dequant_rdna`
# does `tl.static_assert(OUT_DTYPE == tl.float16 or OUT_DTYPE == tl.bfloat16)`
# and the matmul kernel derives that constexpr from `x.dtype`.
_NATIVE_ACT_DTYPES = (torch.bfloat16, torch.float16)


def _is_rdna4_platform() -> bool:
    """True on gfx1200 / gfx1201 only.

    ``on_rdna4()`` reads a module-level amdsmi probe in
    ``vllm/platforms/rocm.py``; it is a plain Python bool, does not initialize
    HIP, and is Dynamo-safe.  Deliberately tighter than the ``on_gfx1x()``
    predicate the out-of-tree 0.19.1 patch used: ``on_gfx1x()`` also admits
    RDNA3 (gfx11), where ``get_rdna_version_host()`` returns 3 and the RDNA4
    tile-hint branch in the graft's ``opt_flags`` never fires; and ``on_gfx12x()``
    would admit future gfx12 parts this has never run on.
    """
    if not current_platform.is_rocm():
        return False

    from vllm.platforms.rocm import on_rdna4

    return on_rdna4()


@cache
def _has_rdna_mxfp4_graft() -> tuple[bool, str | None]:
    """Whether the installed ``triton_kernels`` carries the RDNA MXFP4 graft.

    Stock ``triton_kernels`` has neither ``RDNAMXValueLayout`` nor
    ``mxfp4_dequant_rdna``; without them ``matmul_ogs`` falls through to the
    strided / CDNA branch, which on gfx12 is not merely slow — it would decode
    the weights with the wrong shape assumptions.  Call this only after the
    arch predicate has passed, so non-RDNA4 hosts never import the package.
    """
    # NOTE: has_triton_kernels() also *imports* the package (it calls
    # import_triton_kernels(), which aliases vllm.third_party.triton_kernels
    # into sys.modules when no wheel is installed).  It must therefore run
    # before any `from triton_kernels...` import below, and it must not run at
    # module scope.
    if not has_triton_kernels():
        return False, "the triton_kernels package is not available"

    try:
        from triton_kernels.target_info import (  # noqa: F401
            get_rdna_version_host,
        )
        from triton_kernels.tensor_details.layout import (  # noqa: F401
            RDNAMXValueLayout,
        )
        from triton_kernels.tensor_details.layout_details.rdna_value import (
            mxfp4_dequant_rdna,  # noqa: F401
        )
    except ImportError:
        return False, (
            "the installed triton_kernels does not carry the RDNA MXFP4 graft "
            "(RDNAMXValueLayout / mxfp4_dequant_rdna / get_rdna_version_host)"
        )

    return True, None


def _dequant_mxfp4_reference(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Pure-torch MXFP4 dequantization, used by the explicit fallback path.

    ``weight`` is uint8 ``[N, K // 2]`` (low nibble = even/lower K index),
    ``weight_scale`` is E8M0 uint8 ``[N, K // 32]``; the dequantized value is
    ``LUT[nibble] * 2 ** (scale_u8 - 127)``.

    Deliberately does not call ``torch.ops.vllm.dequant_mxfp4``: that op needs
    the optional ``amd-quark`` package, and this fallback has to stay reachable
    whenever the fast path declines.  Algorithm copied from the operator's
    0.19.1 RDNA4 patch (``_dequant_mxfp4_to_dtype``).
    """
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=weight.device)

    lo = (weight & 0x0F).long()
    hi = ((weight >> 4) & 0x0F).long()
    w = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(
        weight.shape[0], weight.shape[1] * 2
    )

    # E8M0: 2 ** (byte - 127), one scale per group of 32 along K.
    scale = torch.exp2((weight_scale.to(torch.int32) - 127).float())
    scale = scale.repeat_interleave(MXFP4_GROUP_SIZE, dim=-1)
    if scale.shape[-1] > w.shape[-1]:
        scale = scale[:, : w.shape[-1]]

    return (w * scale).to(out_dtype)


# NOTE: no importar triton_kernels a nivel de módulo.  Igual que en aiter.py,
# importarlo temprano puede inicializar HIP y forzar que el engine core haga
# spawn en vez de fork.  `_is_rdna4_platform()` sólo consulta el arch string
# resuelto por amdsmi, así que es HIP-free; los imports reales viven dentro del
# cuerpo del custom op, que sólo corre sobre el device.
if _is_rdna4_platform():
    from vllm.utils.torch_utils import direct_register_custom_op

    def rdna_mxfp4_gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """``x @ dequant(weight).T`` on RDNA4, dequantizing inside the GEMM.

        Args:
            x: 2-D activations ``[M, K]``, bf16 or fp16, row-major.
            weight: packed MXFP4 weights, uint8 ``[N, K // 2]``, row-major
                (contiguous — guaranteed by ``process_weights_after_loading``).
            weight_scale: E8M0 scales, uint8 ``[N, K // 32]``, row-major.

        Returns:
            ``[M, N]`` in ``x.dtype``.  Bias is *not* folded in here: the
            ``matmul_ogs`` bias pointer is indexed by expert and consumed in
            fp32 on the MoE path; the caller adds the bias afterwards, exactly
            like ``AiterMxfp4LinearKernel`` does.
        """
        from triton_kernels.matmul_ogs import (
            FlexCtx,
            PrecisionConfig,
            RoutingData,
            matmul_ogs,
        )
        from triton_kernels.numerics import InFlexData
        from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
        from triton_kernels.tensor_details.layout import (
            RDNAMXValueLayout,
            StridedLayout,
        )

        assert x.ndim == 2, "rdna_mxfp4_gemm expects 2-D activations"

        # Same recipe as `_swizzle_mxfp4` in
        # layers/quantization/utils/mxfp4_utils.py (the MoE weight path): the
        # quantization axis must end up on dim 1, so BOTH tensors are handed
        # over as transposed VIEWS, never as `.T.contiguous()` copies.
        #
        # Why the view matters:
        #   * `wrap_torch_tensor(..., dtype=FP4)` doubles the dimension whose
        #     stride is 1.  weight [N, K//2] (strides (K//2, 1)) transposes to
        #     [K//2, N] with strides (1, K//2), so dim -2 is doubled and the
        #     logical shape becomes [K, N] — which is what matmul_ogs wants
        #     (it reads `K_W, N = w.shape[-2:]` and asserts `K == K_W`).
        #   * matmul_ogs asserts `w.stride(-2) == 1` whenever the value layout
        #     is not StridedLayout.  A `.contiguous()` here would break exactly
        #     that assert.
        #   * for the scales the kernel indexes
        #     `WMxScale + k_idx*stride(-2) + n_idx*stride(-1)`, so the scale
        #     tensor must be logically [K//32, N]; the transposed view of the
        #     row-major [N, K//32] checkpoint tensor gives strides (1, K//32),
        #     which is the correct addressing AND keeps the K axis contiguous
        #     for the [BLOCK_N, MX_SCALE_BLOCK_K] tile load.
        w_t = weight.transpose(-2, -1)
        s_t = weight_scale.transpose(-2, -1)

        # RDNAMXValueLayout is a no-op swizzle (`swizzle_data` is the
        # identity).  Its only job is to carry name="RDNA_VALUE" into the
        # kernel as SWIZZLE_MX_VALUE, which selects the `mxfp4_dequant_rdna`
        # branch (plain `tl.dot`, no `tl.dot_scaled`, no MFMA, no inline asm).
        # Scales stay strided (StridedLayout.name is None -> flat load).
        w_tensor = convert_layout(wrap_torch_tensor(w_t, dtype=FP4), RDNAMXValueLayout)
        s_tensor = convert_layout(wrap_torch_tensor(s_t), StridedLayout)

        precision_config = PrecisionConfig(
            weight_scale=s_tensor,
            flex_ctx=FlexCtx(rhs_data=InFlexData()),
        )

        # RoutingData explícito, NO None — esto es rendimiento, no cosmética.
        # Con `routing_data=None`, matmul_ogs sustituye
        # `RoutingData(None, None, max(1, w.shape[0]), 1)`; para un peso denso
        # `w.shape[0]` es K, así que opt_flags calcula
        # `tokens_per_expt = max(1, M // K) == 1` y clava `block_m = 16` para
        # toda M.  El resultado es idéntico bit a bit, pero mucho más lento en
        # M grande.  Con n_expts_tot=1, `tokens_per_expt == M`.
        routing_data = RoutingData(None, None, 1, 1)

        y = matmul_ogs(
            x,
            w_tensor,
            None,
            routing_data=routing_data,
            precision_config=precision_config,
        )
        # out_dtype defaults to x.dtype inside matmul_ogs; convert defensively
        # rather than let a dtype surprise propagate into the residual stream.
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y

    def rdna_mxfp4_gemm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            (*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device
        )

    # matmul_ogs is host-side Python that inspects device properties and
    # computes launch flags; Dynamo cannot trace it.  Wrapping it in a custom
    # op with a fake impl keeps the layer torch.compile-safe (aiter.py
    # precedent).
    #
    # The op takes plain torch tensors and re-wraps them into
    # triton_kernels.Tensor on every call.  The wrap is pure host-side
    # bookkeeping (RDNAMXValueLayout.swizzle_data is the identity, so no data
    # moves), and whether a triton_kernels.Tensor may legally cross a custom-op
    # boundary is unverified.
    # TODO(rdna-mxfp4): if profiling ever shows the per-call wrap to be
    # material, cache the wrapped tensors ON THE LAYER (not on `self` — one
    # kernel instance serves many layers) and confirm the closure survives
    # Dynamo / cudagraph capture before doing so.
    direct_register_custom_op(
        op_name="rdna_mxfp4_gemm",
        op_func=rdna_mxfp4_gemm,
        mutates_args=[],
        fake_impl=rdna_mxfp4_gemm_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class RdnaMxfp4LinearKernel(MxFp4LinearKernel):
    """Weight-only (A16) MXFP4 GEMM for RDNA4 via the triton_kernels RDNA graft.

    Weights stay packed at 4 bits in VRAM; each tile is dequantized to the
    activation dtype inside the Triton GEMM.  Compared with
    ``EmulationMxfp4LinearKernel`` this removes a transient full-size
    high-precision dequant buffer per forward; the primary win is memory
    traffic, not necessarily raw latency on a single small layer.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Arch first: on anything that is not gfx1200/gfx1201 we return before
        # touching triton_kernels at all.  `compute_capability` is ignored — it
        # is an SM-style number and carries no RDNA information.
        if not current_platform.is_rocm():
            return False, "not running on ROCm"

        if not _is_rdna4_platform():
            return False, "requires an AMD RDNA4 GPU (gfx1200/gfx1201)"

        has_graft, reason = _has_rdna_mxfp4_graft()
        if not has_graft:
            from vllm.model_executor.kernels.linear import _get_linear_backend

            # Only nag when this kernel was actually in the running: staying
            # quiet when the user disabled it or pinned another backend
            # mirrors AiterMxfp4LinearKernel.
            if (
                cls.__name__ not in envs.VLLM_DISABLED_KERNELS
                and _get_linear_backend() == "auto"
            ):
                logger.warning_once(
                    "This GPU is RDNA4 and could run MXFP4 GEMMs natively "
                    "through the triton_kernels RDNA value layout, but %s. "
                    "Falling back to MXFP4 emulation (weights dequantized to "
                    "high precision on every forward).",
                    reason,
                )
            return False, reason

        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        # `CompressedTensorsW4A4Mxfp4` always requests kMxfp4Dynamic, even for
        # weight-only checkpoints, so accepting-and-ignoring it is the only way
        # this kernel is ever selected.  Exactly the precedent set by
        # MarlinMxFp4LinearKernel and HummingMxFp4LinearKernel (same wording).
        #
        # CONSECUENCIA NUMÉRICA, explícita: cuando el config pide
        # kMxfp4Dynamic, EmulationMxfp4LinearKernel además hace QDQ de la
        # activación (`quant_dequant_mxfp4(x)`); este kernel NO.  El resultado
        # es MÁS preciso, no menos, pero NO es bit-idéntico al camino que hoy
        # corre en producción.  Cualquier A/B contra la emulación tiene que
        # comparar contra `dequant_mxfp4 + F.linear` SIN el QDQ, o la
        # diferencia se atribuirá al kernel por error.
        if config.activation_quant_key not in (None, kMxfp4Dynamic):
            return False, "only supports MXFP4 dynamic or unquantized activations"
        if config.activation_quant_key is not None:
            logger.warning_once(
                "RdnaMxfp4LinearKernel is a weight-only (A16) kernel; "
                "the requested activation quantization (%s) is ignored.",
                config.activation_quant_key,
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # On entry CompressedTensorsW4A4Mxfp4 has already rebound
        # layer.weight_packed -> layer.weight, so:
        #   layer.weight        uint8 [N, K // 2]   (K = input_size_per_partition)
        #   layer.weight_scale  uint8 [N, K // 32]  (E8M0)
        # No repacking is needed — both the fast path and the fallback consume
        # exactly this layout.  We only force contiguity (so that
        # `stride().index(1)` inside `wrap_torch_tensor` is unambiguous and the
        # transposed view really is column-major) and rebind as plain
        # Parameters, as the sibling kernels do.
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data

        # Validation uses exceptions, not `assert`: `python -O` strips asserts,
        # and every one of these mistakes would otherwise decode silently into
        # plausible-but-wrong weights.
        if weight.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
            raise ValueError(
                "RdnaMxfp4LinearKernel expects packed uint8 weights and uint8 "
                f"E8M0 scales, got {weight.dtype} / {weight_scale.dtype}"
            )
        if weight.ndim != 2 or weight_scale.ndim != 2:
            raise ValueError(
                "RdnaMxfp4LinearKernel expects 2-D weights and scales, got "
                f"{weight.ndim}-D / {weight_scale.ndim}-D"
            )

        n, packed_k = weight.shape
        k = packed_k * 2
        # K here is already TP-sharded.  compressed-tensors guarantees
        # group_size=32, but a bad shard would silently shift scale indexing.
        if k % MXFP4_GROUP_SIZE != 0:
            raise ValueError(
                f"RdnaMxfp4LinearKernel needs K ({k}) to be a multiple of "
                f"{MXFP4_GROUP_SIZE} after tensor-parallel sharding"
            )
        if tuple(weight_scale.shape) != (n, k // MXFP4_GROUP_SIZE):
            raise ValueError(
                f"unexpected MXFP4 scale shape {tuple(weight_scale.shape)}, "
                f"expected {(n, k // MXFP4_GROUP_SIZE)}"
            )

        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.contiguous(), requires_grad=False)

        # Belt and braces.  `on_rdna4()` reads the amdsmi arch string; the
        # graft's `opt_flags` gates its RDNA4 WMMA v2 tile hints on
        # `get_rdna_version_host()`, which reads torch.cuda device properties.
        # That call cannot live at module scope or in `is_supported` (it would
        # initialize HIP), but by the time weights are on the device HIP is up.
        # If the two ever disagree, degrade to the correct-but-slow path rather
        # than crash a live engine.
        from triton_kernels.target_info import get_rdna_version_host

        rdna_version = get_rdna_version_host()
        layer.rdna_mxfp4_native = rdna_version == 4
        if not layer.rdna_mxfp4_native:
            logger.warning_once(
                "vllm.platforms.rocm reports RDNA4 but "
                "triton_kernels.get_rdna_version_host() returned %s; using the "
                "dequantize + high-precision linear fallback for this layer.",
                rdna_version,
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, packed_k = layer.weight.shape
        k = packed_k * 2
        out_shape = (*x.shape[:-1], n)

        if x.shape[-1] != k:
            raise ValueError(
                f"activation K ({x.shape[-1]}) does not match the packed "
                f"weight K ({k})"
            )

        use_native = getattr(layer, "rdna_mxfp4_native", False) and (
            x.dtype in _NATIVE_ACT_DTYPES
        )

        if not use_native:
            # Explicit fallback: same numbers, no graft involved.  Reached for
            # fp32 activations (the in-kernel dequant only emits fp16/bf16) and
            # when the RDNA4 device probe disagreed with the platform
            # predicate.  Costs a transient high-precision weight, i.e. it is
            # emulation — correct, just not the point of this kernel.
            dq_w = _dequant_mxfp4_reference(layer.weight, layer.weight_scale, x.dtype)
            return F.linear(x, dq_w, bias)

        # matmul_ogs treats a 3-D x as a *batched* matmul and then asserts
        # `w.ndim == 3`, so N-D activations must be flattened here (F.linear,
        # the oracle, handles them itself).  Flatten by x's own last dim — the
        # K check above already tied it to the weight, and reshaping by a
        # mismatched K could otherwise succeed with a wrong M.
        x_2d = x.reshape(-1, x.shape[-1])

        if x_2d.shape[0] == 0:
            # Empty batch: don't enter the kernel at all.  numel == 0, so the
            # uninitialized buffer carries no garbage.
            return torch.empty(out_shape, dtype=x.dtype, device=x.device)

        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()

        y = torch.ops.vllm.rdna_mxfp4_gemm(x_2d, layer.weight, layer.weight_scale)

        if bias is not None:
            y = y + bias

        return y.reshape(out_shape)


# ── Cabos sueltos / TODOs ────────────────────────────────────────────────────
#
# 1. TODO(rdna-mxfp4) — VALIDACIÓN NUMÉRICA OBLIGATORIA ANTES DE HACER SHIP.
#    El parche RDNA4 0.19.1 del operador usaba `RDNAMXValueLayout` SÓLO en la
#    ruta MoE (`_swizzle_mxfp4` -> gpt_oss_triton_kernels_moe); su camino denso
#    W4A16 dequantizaba en tiempo de carga y llamaba a F.linear.  O sea: el
#    cableado denso de este archivo NO tiene precedente probado.  Antes de
#    promocionarlo hay que comparar, capa por capa, contra
#    `F.linear(x, dequant_mxfp4(w, s, x.dtype), bias)` (sin QDQ de activación).
#    Y ojo con el modo de fallo peligroso: si las dos transposiciones se
#    intercambiaran, la forma de salida seguiría siendo correcta en toda
#    proyección CUADRADA (q/k/v/o de un modelo con head_dim uniforme), así que
#    NINGUNA comprobación de shape lo detecta — sólo el test numérico.
#    Tolerancia: en bf16 la igualdad exacta se da a M pequeño; a M grande
#    aparecen diferencias del orden de 1 ULP por el orden de acumulación.  Usar
#    rtol ~1e-2 en bf16, o comparar en fp32 con atol = y.abs().max() * 3e-3.
#
# 2. TODO(rdna-mxfp4) — el contrato de `matmul_ogs` que asume este archivo
#    (w column-major, escala lógica [K//32, N], RoutingData explícito,
#    `is_persistent=False` en la rama RDNA4 de opt_flags) se verificó leyendo
#    una COPIA de `triton_kernels/matmul_ogs.py`, no la wheel exacta que el
#    Dockerfile injerta en site-packages.  Re-verificar contra la wheel
#    instalada antes de confiar en él; un cambio upstream en el orden de
#    `w_scale_strides` (hoy: e, k, n = strides[-3:]) rompería las escalas en
#    silencio.
#
# 3. Numérica bf16 (por qué se espera igualdad con el oráculo): los valores
#    E2M1 (0.5 … 6.0) y las escalas E8M0 (potencias exactas de dos) son
#    representables exactamente en bf16, así que el `w_dequant * scale` que
#    hace el kernel en bf16 coincide con el `(w * scale_fp32).to(bf16)` de la
#    referencia.  La acumulación es fp32 en ambos casos.
#
# 4. Valores límite de E8M0 (documentados, NO "arreglados" en silencio):
#    `mxfp4_dequant_rdna` convierte el byte 0x00 a bf16 0.0 y 0xFF a +inf,
#    mientras que la referencia fp32 da 2^-127 y un exponente saturado.  Con
#    OUT_DTYPE=fp16 además satura el exponente al rango [0, 30], o sea que las
#    escalas extremas se recortan.  No ocurre en checkpoints reales de
#    compressed-tensors; si algún día ocurre, se verá como degradación muda.
#
# 5. Rendimiento: el margen de tuning vive en la rama RDNA4 de `opt_flags` del
#    graft (block_n, num_stages, split_k para M flaca), NO aquí.  NO llamar a
#    `update_opt_flags_constraints` desde este archivo: es estado global del
#    proceso y se filtraría a la ruta MoE.
#
# 6. TODO(rdna-mxfp4) — sin verificar bajo captura de cudagraph.  El custom op
#    con `fake_impl` es lo que exige el precedente de aiter.py, pero la ruta
#    completa (matmul_ogs + autotune del graft dentro de un grafo capturado) no
#    se ha probado en este hardware.