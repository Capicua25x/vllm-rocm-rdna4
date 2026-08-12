#!/usr/bin/env python3
"""
Lever-A go/no-go: does head-dim CHUNKING of the head-512 QK dot reduce Triton's LDS
(shared-mem) allocation on gfx1201, so a larger TILE_SIZE fits in RDNA4's 64 KiB?

Reproduces the real kernel's inner tile (QK over head=512 + softmax + PV out=512). Two modes:
  - CHUNK=0 : monolithic  S = tl.dot(Q, K)   over the full 512 head dim
  - CHUNK>0 : loop the 512 contraction in CHUNK-wide slices  S += tl.dot(Q[:,d0:], K[d0:,:])
PV is left monolithic in BOTH modes (isolates the QK-chunk effect).
Report Triton's compiled shared-mem bytes (metadata.shared) vs the 65536 limit.
Known real-kernel crash: monolithic @ TILE>=64 -> "131072 > 65536".
GO = QK-chunking drops shared below 65536 where monolithic exceeds it.
"""
import torch, triton, triton.language as tl

BLOCK_M = 128
HEAD = 512  # Gemma-4 full-attention head_dim

@triton.jit
def attn_tile(Q, K, V, Out, n_kv,
              BLOCK_M: tl.constexpr, HEAD: tl.constexpr, TILE: tl.constexpr,
              CHUNK: tl.constexpr):
    qi = tl.arange(0, BLOCK_M)
    di = tl.arange(0, HEAD)
    ti = tl.arange(0, TILE)
    acc = tl.zeros((BLOCK_M, HEAD), dtype=tl.float32)
    M = tl.full((BLOCK_M,), -1e30, tl.float32)
    L = tl.zeros((BLOCK_M,), tl.float32)
    if CHUNK == 0:
        q = tl.load(Q + qi[:, None] * HEAD + di[None, :]).to(tl.float16)   # full (BLOCK_M, HEAD)
    for j in range(0, n_kv):
        v = tl.load(V + ti[:, None] * HEAD + di[None, :]).to(tl.float16)   # (TILE, HEAD)
        if CHUNK > 0:
            # load Q/K in CHUNK-wide head slices so no full-512 operand is ever staged
            s = tl.zeros((BLOCK_M, TILE), tl.float32)
            for d0 in tl.static_range(0, HEAD, CHUNK):
                dc = d0 + tl.arange(0, CHUNK)
                q_c = tl.load(Q + qi[:, None] * HEAD + dc[None, :]).to(tl.float16)  # (BLOCK_M, CHUNK)
                k_c = tl.load(K + dc[:, None] * TILE + ti[None, :]).to(tl.float16)  # (CHUNK, TILE)
                s += tl.dot(q_c, k_c)
        else:
            k = tl.load(K + di[:, None] * TILE + ti[None, :]).to(tl.float16)   # (HEAD, TILE)
            s = tl.dot(q, k)
        m_j = tl.maximum(M, tl.max(s, axis=1))
        p = tl.exp(s - m_j[:, None])
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + tl.sum(p, axis=1)
        M = m_j
        acc += tl.dot(p.to(tl.float16), v)            # PV monolithic (both modes)
    acc = acc / L[:, None]
    tl.store(Out + qi[:, None] * HEAD + di[None, :], acc)


def probe(tile, chunk):
    dev = "cuda"
    q = torch.randn(BLOCK_M, HEAD, device=dev, dtype=torch.float16)
    k = torch.randn(HEAD, tile, device=dev, dtype=torch.float16)
    v = torch.randn(tile, HEAD, device=dev, dtype=torch.float16)
    out = torch.empty(BLOCK_M, HEAD, device=dev, dtype=torch.float32)
    try:
        compiled = attn_tile.warmup(q, k, v, out, 1,
                                    BLOCK_M=BLOCK_M, HEAD=HEAD, TILE=tile, CHUNK=chunk,
                                    grid=(1,))
        shared = compiled.metadata.shared
        tag = "OK (<=64KiB)" if shared <= 65536 else "OVER 64KiB"
        print(f"  TILE={tile:<3} CHUNK={chunk:<3} -> shared={shared:>7} B  {tag}", flush=True)
        return shared
    except Exception as e:
        msg = " | ".join(str(e).split('\n'))[:300]
        print(f"  TILE={tile:<3} CHUNK={chunk:<3} -> FAIL: {msg}", flush=True)
        return None


if __name__ == "__main__":
    print(f"gfx: {torch.cuda.get_device_name(0)}", flush=True)
    print("== MONOLITHIC QK (CHUNK=0) ==", flush=True)
    for t in (32, 64, 128):
        probe(t, 0)
    print("== CHUNKED QK (CHUNK=128) ==", flush=True)
    for t in (32, 64, 128):
        probe(t, 128)
    print("== CHUNKED QK (CHUNK=64) ==", flush=True)
    for t in (64, 128):
        probe(t, 64)
