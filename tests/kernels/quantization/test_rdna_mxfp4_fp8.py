# SPDX-License-Identifier: Apache-2.0
"""RDNA4-only: RdnaMxfp4Fp8LinearKernel vs reference dequant, all regimes + odd shapes.
Run inside the RDNA4 image: python tests/kernels/quantization/test_rdna_mxfp4_fp8.py"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from _rdna_mxfp4_ref import make_mxfp4_weight, dequant_ref
from vllm.model_executor.kernels.linear.mxfp4.rdna_fp8 import RdnaMxfp4Fp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig

def main():
    dev = "cuda"; g = torch.Generator(device=dev); g.manual_seed(0)
    k = RdnaMxfp4Fp8LinearKernel(MxFp4LinearLayerConfig())
    for (N, K) in [(24, 5120), (100, 3072), (8704, 5120), (5120, 8704)]:
        wp, ws = make_mxfp4_weight(N, K, dev, g); wdq = dequant_ref(wp, ws)
        class L(torch.nn.Module): pass
        l = L(); l.weight = torch.nn.Parameter(wp, requires_grad=False); l.weight_scale = torch.nn.Parameter(ws, requires_grad=False)
        k.process_weights_after_loading(l)
        for M in [1, 5, 64, 129, 300, 600, 2048]:
            x = (torch.randn(M, K, generator=g, device=dev) * 0.5); x[:, ::97] *= 20; x = x.to(torch.bfloat16)
            y = k.apply_weights(l, x, None).float(); ref = x.float() @ wdq.T
            rel = ((y - ref).norm() / ref.norm()).item()
            tol = 2e-2 if M <= 128 else 4e-3    # fused e4m3 path vs bf16-activation paths
            assert rel < tol, (N, K, M, rel)
            print(f"ok N={N} K={K} M={M} rel={rel:.2e}")
    print("PASS")

if __name__ == "__main__":
    main()
