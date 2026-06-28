#!/usr/bin/env python3
"""Prototype + de-risk the head-512 flash rewrite's core mechanic BEFORE touching the real kernel.

Validates the Triton "list-of-chunk-tensors" pattern that the design needs (Triton can't slice a loaded
tensor or write a tensor slice), for BOTH:
  - QK: Q pre-loaded as NUM_CHUNKS head-slices (resident, loaded once); S accumulated over head-chunks.
  - PV: acc held as NUM_CHUNKS output-head accumulators; each updated per out-chunk (V never fully staged).
Online softmax across KV tiles, exactly like kernel_unified_attention_2d.

Checks: (1) CHUNK output == MONO output == torch reference (correctness of the decomposition);
        (2) Triton compiled shared-mem (metadata.shared): CHUNK < MONO (the occupancy lever).
Full (non-causal) attention — the chunk math is mask-independent, so masking is omitted for clarity.
"""
import torch, triton, triton.language as tl

BM = 16      # BLOCK_M (Gemma full-attn: BLOCK_Q=2 * num_queries_per_kv=8)
HEAD = 512   # global_head_dim
TILE = 32    # KV positions per iter (nbatch_fa)
NKV = 256    # total keys (8 tiles)


@triton.jit
def attn_mono(Qp, Kp, Vp, Op, scale,
              BM: tl.constexpr, HEAD: tl.constexpr, TILE: tl.constexpr, NKV: tl.constexpr):
    qi = tl.arange(0, BM)
    d = tl.arange(0, HEAD)
    t = tl.arange(0, TILE)
    Q = tl.load(Qp + qi[:, None] * HEAD + d[None, :])            # [BM, HEAD] resident
    M = tl.full((BM,), -1e30, tl.float32)
    L = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, HEAD), tl.float32)
    for j in range(0, NKV // TILE):
        n = j * TILE + t
        K = tl.load(Kp + d[:, None] * 1 + n[None, :] * HEAD)     # [HEAD, TILE]
        V = tl.load(Vp + n[:, None] * HEAD + d[None, :])         # [TILE, HEAD]
        S = scale * tl.dot(Q, K)                                  # [BM, TILE]
        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.exp(S - m_j[:, None])
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None] + tl.dot(P.to(tl.float16), V)
        L = L * alpha + tl.sum(P, axis=1)
        M = m_j
    acc = acc / L[:, None]
    tl.store(Op + qi[:, None] * HEAD + d[None, :], acc)


@triton.jit
def _pv(acc_c, alpha, Ph, Vp, n, HEAD: tl.constexpr, KC: tl.constexpr, c: tl.constexpr):
    # online-softmax PV update for output-head chunk c: only a [TILE,KC] V-chunk is staged.
    dc = tl.arange(0, KC)
    Vc = tl.load(Vp + n[:, None] * HEAD + (c * KC + dc)[None, :])   # [TILE, KC]
    return acc_c * alpha[:, None] + tl.dot(Ph, Vc)


@triton.jit
def attn_chunk(Qp, Kp, Vp, Op, scale,
               BM: tl.constexpr, HEAD: tl.constexpr, TILE: tl.constexpr, NKV: tl.constexpr,
               KC: tl.constexpr):
    # NUM = HEAD//KC hardcoded to 8 (Gemma head=512, KC=64). Explicit 8-tuple accumulators — the only
    # Triton-3.6 construct that carries N per-chunk accumulators across the real KV loop (probe-verified:
    # generator-tuples / tl.cat / list-literals all fail; explicit tuple literal + constexpr index works).
    qi = tl.arange(0, BM)
    t = tl.arange(0, TILE)
    dc = tl.arange(0, KC)
    NUM: tl.constexpr = HEAD // KC
    M = tl.full((BM,), -1e30, tl.float32)
    L = tl.zeros((BM,), tl.float32)
    accs = (tl.zeros((BM, KC), tl.float32), tl.zeros((BM, KC), tl.float32),
            tl.zeros((BM, KC), tl.float32), tl.zeros((BM, KC), tl.float32),
            tl.zeros((BM, KC), tl.float32), tl.zeros((BM, KC), tl.float32),
            tl.zeros((BM, KC), tl.float32), tl.zeros((BM, KC), tl.float32))
    for j in range(0, NKV // TILE):
        n = j * TILE + t
        # QK: accumulate S over head-chunks; only a [KC, TILE] K-chunk staged at a time. Q-chunk re-loaded
        # per chunk (L2-cached, tiny at BM=16) — avoids needing a persistent Q tuple.
        S = tl.zeros((BM, TILE), tl.float32)
        for c in range(NUM):
            Qc = tl.load(Qp + qi[:, None] * HEAD + (c * KC + dc)[None, :])      # [BM, KC]
            Kc = tl.load(Kp + (c * KC + dc)[:, None] * 1 + n[None, :] * HEAD)   # [KC, TILE]
            S += tl.dot(Qc, Kc)
        S = S * scale
        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.exp(S - m_j[:, None])
        alpha = tl.exp(M - m_j)
        L = L * alpha + tl.sum(P, axis=1)
        M = m_j
        Ph = P.to(tl.float16)
        # PV: explicit 8-tuple update; each _pv stages only a [TILE,KC] V-chunk (V never fully resident)
        accs = (_pv(accs[0], alpha, Ph, Vp, n, HEAD, KC, 0),
                _pv(accs[1], alpha, Ph, Vp, n, HEAD, KC, 1),
                _pv(accs[2], alpha, Ph, Vp, n, HEAD, KC, 2),
                _pv(accs[3], alpha, Ph, Vp, n, HEAD, KC, 3),
                _pv(accs[4], alpha, Ph, Vp, n, HEAD, KC, 4),
                _pv(accs[5], alpha, Ph, Vp, n, HEAD, KC, 5),
                _pv(accs[6], alpha, Ph, Vp, n, HEAD, KC, 6),
                _pv(accs[7], alpha, Ph, Vp, n, HEAD, KC, 7))
    Li = (1.0 / L)[:, None]
    tl.store(Op + qi[:, None] * HEAD + (0 * KC + dc)[None, :], accs[0] * Li)
    tl.store(Op + qi[:, None] * HEAD + (1 * KC + dc)[None, :], accs[1] * Li)
    tl.store(Op + qi[:, None] * HEAD + (2 * KC + dc)[None, :], accs[2] * Li)
    tl.store(Op + qi[:, None] * HEAD + (3 * KC + dc)[None, :], accs[3] * Li)
    tl.store(Op + qi[:, None] * HEAD + (4 * KC + dc)[None, :], accs[4] * Li)
    tl.store(Op + qi[:, None] * HEAD + (5 * KC + dc)[None, :], accs[5] * Li)
    tl.store(Op + qi[:, None] * HEAD + (6 * KC + dc)[None, :], accs[6] * Li)
    tl.store(Op + qi[:, None] * HEAD + (7 * KC + dc)[None, :], accs[7] * Li)


def run(kernel, KC=None):
    dev = "cuda"
    torch.manual_seed(0)
    Q = torch.randn(BM, HEAD, device=dev, dtype=torch.float16)
    K = torch.randn(NKV, HEAD, device=dev, dtype=torch.float16)
    V = torch.randn(NKV, HEAD, device=dev, dtype=torch.float16)
    O = torch.empty(BM, HEAD, device=dev, dtype=torch.float32)
    scale = 1.0 / (HEAD ** 0.5)
    if KC is None:
        c = kernel[(1,)](Q, K, V, O, scale, BM=BM, HEAD=HEAD, TILE=TILE, NKV=NKV)
    else:
        c = kernel[(1,)](Q, K, V, O, scale, BM=BM, HEAD=HEAD, TILE=TILE, NKV=NKV, KC=KC)
    ref = torch.softmax((Q.float() @ K.float().T) * scale, dim=1) @ V.float()   # [BM, HEAD]
    return O, ref, c.metadata.shared


if __name__ == "__main__":
    print(f"gfx: {torch.cuda.get_device_name(0)}", flush=True)
    O_m, ref, shared_m = run(attn_mono)
    err_m = (O_m - ref).abs().max().item()
    print(f"MONO    : shared={shared_m:>7} B  max|out-ref|={err_m:.4e}", flush=True)
    for KC in (64,):
        O_c, ref, shared_c = run(attn_chunk, KC=KC)
        err_c = (O_c - ref).abs().max().item()
        err_mc = (O_c - O_m).abs().max().item()
        verdict = "OK" if (err_c < 5e-2 and shared_c < shared_m) else "CHECK"
        print(f"CHUNK KC={KC:<3}: shared={shared_c:>7} B  max|out-ref|={err_c:.4e}  max|chunk-mono|={err_mc:.4e}  "
              f"LDS {shared_m}->{shared_c} ({100*(1-shared_c/shared_m):.0f}% less)  {verdict}", flush=True)
