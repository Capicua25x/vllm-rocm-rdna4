#!/usr/bin/env python3
"""GQA-split scoping probe: map the LDS-feasible (BLOCK_M, TILE) space for the head-512
full-attention tile on gfx1201. The GQA-split rewrite shrinks BLOCK_M (split the 128-head
GQA group across G programs) so a larger TILE fits. But shrinking BLOCK_M moves the binding
LDS term from the Q operand (BLOCK_M x 512) to the K operand (512 x TILE) — this sweep finds
the real ceiling so the rewrite targets a TILE that actually compiles.

Monolithic dot (NO head-dim chunk — chunking was refuted 2026-06-27). Reports Triton's
compiled shared-mem bytes vs RDNA4's 65536 limit. Compile-only (warmup), no launch.
"""
import torch, triton, triton.language as tl

HEAD = 512  # Gemma-4 full-attention head_dim

@triton.jit
def attn_tile(Q, K, V, Out, n_kv,
              BLOCK_M: tl.constexpr, HEAD: tl.constexpr, TILE: tl.constexpr):
    qi = tl.arange(0, BLOCK_M)
    di = tl.arange(0, HEAD)
    ti = tl.arange(0, TILE)
    acc = tl.zeros((BLOCK_M, HEAD), dtype=tl.float32)
    M = tl.full((BLOCK_M,), -1e30, tl.float32)
    L = tl.zeros((BLOCK_M,), tl.float32)
    q = tl.load(Q + qi[:, None] * HEAD + di[None, :]).to(tl.float16)   # (BLOCK_M, HEAD) resident, loaded once
    for j in range(0, n_kv):
        k = tl.load(K + di[:, None] * TILE + ti[None, :]).to(tl.float16)   # (HEAD, TILE)
        v = tl.load(V + ti[:, None] * HEAD + di[None, :]).to(tl.float16)   # (TILE, HEAD)
        s = tl.dot(q, k)                                                   # (BLOCK_M, TILE)
        m_j = tl.maximum(M, tl.max(s, axis=1))
        p = tl.exp(s - m_j[:, None])
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + tl.sum(p, axis=1)
        M = m_j
        acc += tl.dot(p.to(tl.float16), v)
    acc = acc / L[:, None]
    tl.store(Out + qi[:, None] * HEAD + di[None, :], acc)


def probe(block_m, tile):
    dev = "cuda"
    q = torch.randn(block_m, HEAD, device=dev, dtype=torch.float16)
    k = torch.randn(HEAD, tile, device=dev, dtype=torch.float16)
    v = torch.randn(tile, HEAD, device=dev, dtype=torch.float16)
    out = torch.empty(block_m, HEAD, device=dev, dtype=torch.float32)
    try:
        compiled = attn_tile.warmup(q, k, v, out, 1,
                                    BLOCK_M=block_m, HEAD=HEAD, TILE=tile, grid=(1,))
        shared = compiled.metadata.shared
        tag = "OK" if shared <= 65536 else "OVER-64KiB"
        print(f"  BLOCK_M={block_m:<4} TILE={tile:<4} -> shared={shared:>7} B  {tag}", flush=True)
        return shared
    except Exception as e:
        msg = " | ".join(str(e).split('\n'))[:200]
        print(f"  BLOCK_M={block_m:<4} TILE={tile:<4} -> FAIL: {msg}", flush=True)
        return None


if __name__ == "__main__":
    print(f"gfx: {torch.cuda.get_device_name(0)}", flush=True)
    print("Monolithic head-512 tile; Q loaded ONCE (resident). Map (BLOCK_M, TILE) vs 64KiB LDS.", flush=True)
    for bm in (128, 64, 32, 16):
        for t in (32, 64, 128, 256):
            probe(bm, t)
