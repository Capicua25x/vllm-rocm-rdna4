import triton
import triton.language as tl
from dataclasses import dataclass
from .base import Layout


MXFP_BLOCK_SIZE = tl.constexpr(32)


@dataclass
class RDNAMXValueLayout(Layout):
    """MXFP4 value layout for RDNA4. No hardware swizzle — routes to manual dequant."""
    name: str = "RDNA_VALUE"

    def __init__(self, shape):
        super().__init__(shape)
        self.name = "RDNA_VALUE"

    def swizzle_data(self, data):
        return data

    def unswizzle_data(self, data):
        return data

    def swizzle_block_shape(self, block_shape):
        return block_shape


@triton.jit
def mxfp4_dequant_rdna(w_packed, w_scales, mx_axis: tl.constexpr, OUT_DTYPE: tl.constexpr):
    """
    Dequantize MXFP4 (E2M1 values + E8M0 scales) to fp16/bf16 using pure Triton ops.
    No inline asm, no MFMA — runs on RDNA4 WMMA v2.

    w_packed: [BLOCK_K // 2, BLOCK_N] uint8
    w_scales: [BLOCK_N, MX_SCALE_BLOCK_K] uint8 (E8M0)
    mx_axis:  1 (K groups on last axis of scales)
    OUT_DTYPE: tl.float16 or tl.bfloat16
    Returns: [BLOCK_K, BLOCK_N] in OUT_DTYPE
    """
    tl.static_assert(OUT_DTYPE == tl.float16 or OUT_DTYPE == tl.bfloat16)
    tl.static_assert(w_packed.dtype == tl.uint8)
    tl.static_assert(mx_axis == 1, "RDNA dequant expects mx_axis=1")

    w = w_packed.trans()

    dst_bias: tl.constexpr = 127 if OUT_DTYPE == tl.bfloat16 else 15
    dst_0p5: tl.constexpr = 16128 if OUT_DTYPE == tl.bfloat16 else 0x3800
    dst_m_bits: tl.constexpr = 7 if OUT_DTYPE == tl.bfloat16 else 10

    em0 = w & 0x07
    em1 = w & 0x70

    x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((w & 0x08).to(tl.uint16) << 12)
    x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((w & 0x80).to(tl.uint16) << 8)

    x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
    x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)

    x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
    x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)

    w_dequant = tl.interleave(x0, x1).to(OUT_DTYPE, bitcast=True)

    if OUT_DTYPE == tl.bfloat16:
        scale = (w_scales.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)
    else:
        scale_exp = w_scales.to(tl.int16) - 112
        zero_i16 = tl.full(scale_exp.shape, 0, dtype=tl.int16)
        max_exp_i16 = tl.full(scale_exp.shape, 30, dtype=tl.int16)
        scale_exp = tl.where(scale_exp < zero_i16, zero_i16, scale_exp)
        scale_exp = tl.where(scale_exp > max_exp_i16, max_exp_i16, scale_exp)
        scale = (scale_exp.to(tl.uint16) << 10).to(tl.float16, bitcast=True)

    scale = scale.expand_dims(mx_axis + 1)
    scale = scale.broadcast_to(scale.shape[:mx_axis + 1] + [MXFP_BLOCK_SIZE] + scale.shape[mx_axis + 2:])
    scale = scale.reshape(w_dequant.shape)

    return (w_dequant * scale).trans()
