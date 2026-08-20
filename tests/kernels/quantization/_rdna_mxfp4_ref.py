# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 weights (E2M1 nibbles + E8M0 block-32 scales) x fp8-e4m3 activations, on RDNA4 FP8 WMMA.
acc += dot(x8[BM,32], w8[32,BN]) * scale[block, n]  ;  y = acc * sx[m]
Weight layout: packed uint8 [N, K//2] row-major (low nibble = even k), scales uint8 [N, K//32]."""
import torch, triton, triton.language as tl

@triton.jit
def _nib_to_e4m3_bits(n):
    # n: uint8 in 0..15 (E2M1). Returns uint8 e4m3fn bit pattern of the same value.
    mag = n & 7
    sign = (n >> 3) & 1
    b = tl.where(mag == 0, 0, tl.where(mag == 1, 0x30, 0x38 + (mag - 2) * 4))
    return (b | (sign << 7)).to(tl.uint8)

@triton.jit
def mxfp4_fp8_gemm_kernel(X8, SX, WP, WS, Y, M, N, K,
                          stride_xm, stride_wn, stride_wsn, stride_ym,
                          BM: tl.constexpr, BN: tl.constexpr, NUM_KB: tl.constexpr, EVEN_M: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, 32)
    okp = tl.arange(0, 16)          # packed k index (16 bytes per 32 values)
    mask_m = om < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, NUM_KB):
        # x tile [BM, 32] fp8
        xp = X8 + om[:, None] * stride_xm + (kb * 32 + ok)[None, :]
        if EVEN_M:
            x = tl.load(xp)
        else:
            x = tl.load(xp, mask=mask_m[:, None], other=0.0)
        # w packed tile [BN, 16] uint8 -> [BN, 32] e4m3 bits -> transposed [32, BN]
        wp = WP + on[:, None] * stride_wn + (kb * 16 + okp)[None, :]
        w = tl.load(wp)                                    # [BN,16] uint8
        lo = _nib_to_e4m3_bits(w & 0x0F)                   # even k
        hi = _nib_to_e4m3_bits((w >> 4) & 0x0F)            # odd k
        w2 = tl.join(lo, hi)                               # [BN,16,2] -> interleave even/odd
        w2 = tl.reshape(w2, (BN, 32))                      # [BN,32] k-major within row
        w8 = w2.to(tl.float8e4nv, bitcast=True)
        w8t = tl.trans(w8)                                 # [32, BN]
        part = tl.dot(x, w8t)                              # [BM,BN] fp32
        # scale for this k-block, per n: E8M0 -> fp32 via bit shift
        s = tl.load(WS + on * stride_wsn + kb).to(tl.int32)
        sf = (s << 23).to(tl.float32, bitcast=True)        # 2^(s-127)
        acc += part * sf[None, :]
    sx = tl.load(SX + om, mask=mask_m, other=0.0)
    y = acc * sx[:, None]
    yp = Y + om[:, None] * stride_ym + on[None, :]
    tl.store(yp, y.to(tl.bfloat16), mask=mask_m[:, None])

def quant_x_fp8(x: torch.Tensor):
    """Per-token dynamic e4m3fn quant: returns (x8, sx) with x ≈ x8 * sx."""
    amax = x.abs().amax(dim=-1).float().clamp(min=1e-12)
    sx = amax / 448.0
    x8 = (x.float() / sx[:, None]).to(torch.float8_e4m3fn)
    return x8, sx

def mxfp4_fp8_gemm(x: torch.Tensor, w_packed: torch.Tensor, w_scale: torch.Tensor, BM=None, BN=64, num_warps=4):
    M, K = x.shape; N = w_packed.shape[0]
    assert K % 32 == 0 and w_packed.shape[1] == K // 2 and w_scale.shape == (N, K // 32)
    x8, sx = quant_x_fp8(x)
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    if BM is None: BM = 16 if M <= 16 else (32 if M <= 32 else 64)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    mxfp4_fp8_gemm_kernel[grid](x8, sx, w_packed, w_scale, y, M, N, K,
                                x8.stride(0), w_packed.stride(0), w_scale.stride(0), y.stride(0),
                                BM=BM, BN=BN, NUM_KB=K // 32, EVEN_M=(M % BM == 0), num_warps=num_warps)
    return y

# ---- reference ----
_LUT = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32)
def dequant_ref(w_packed, w_scale, dtype=torch.float32):
    lut = _LUT.to(w_packed.device)
    lo = (w_packed & 0x0F).long(); hi = ((w_packed >> 4) & 0x0F).long()
    w = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(w_packed.shape[0], w_packed.shape[1]*2)
    scale = torch.exp2((w_scale.to(torch.int32) - 127).float()).repeat_interleave(32, dim=-1)
    return (w * scale).to(dtype)

def make_mxfp4_weight(N, K, device, gen):
    """Random MXFP4 weight: pick nibbles + scales roughly like a real layer (values ~N(0,0.02))."""
    w = torch.randn(N, K, generator=gen, device=device) * 0.02
    blk = w.reshape(N, K//32, 32)
    amax = blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    e = torch.floor(torch.log2(amax / 6.0)).clamp(-127, 127)          # scale so max maps to <=6
    scale_u8 = (e + 127).to(torch.uint8).squeeze(-1)                     # [N,K/32]
    q = blk / torch.exp2(e)                                              # values in [-6,6]
    # nearest E2M1 value
    lut = _LUT[:8].to(device)
    idx = (q.abs().unsqueeze(-1) - lut).abs().argmin(dim=-1)             # 0..7
    nib = idx + 8 * (q < 0).to(torch.long)
    nib = nib.reshape(N, K)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), scale_u8.contiguous()
