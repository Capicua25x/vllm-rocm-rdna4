#!/usr/bin/env python3
"""Tiny probe: learn which Triton constructs this build accepts for the head-chunk kernel.
Tests, each isolated so one failure doesn't hide the others:
  A. generator-as-tuple comprehension + constexpr indexing:  Qcs = (load() for c in range(N)); Qcs[c]
  B. tl.cat of two [BM,KC] tensors along the last axis -> [BM, 2*KC]
  C. hardcoded N=8 explicit accumulators (fallback, always works) -- sanity only
"""
import torch, triton, triton.language as tl
BM, KC = 16, 64

@triton.jit
def k_genidx(Xp, Op, BM: tl.constexpr, N: tl.constexpr, KC: tl.constexpr):
    qi = tl.arange(0, BM); dc = tl.arange(0, KC)
    Xs = (tl.load(Xp + qi[:, None] * (N * KC) + (c * KC + dc)[None, :]) for c in range(N))
    acc = tl.zeros((BM, KC), tl.float32)
    for c in range(N):
        acc += Xs[c]
    tl.store(Op + qi[:, None] * KC + dc[None, :], acc)

@triton.jit
def k_cat(Xp, Op, BM: tl.constexpr, KC: tl.constexpr):
    qi = tl.arange(0, BM); dc = tl.arange(0, KC)
    a = tl.load(Xp + qi[:, None] * (2 * KC) + dc[None, :])
    b = tl.load(Xp + qi[:, None] * (2 * KC) + (KC + dc)[None, :])
    catd = tl.cat(a, b, can_reorder=False)             # want [BM, 2*KC]
    qi2 = tl.arange(0, BM); d2 = tl.arange(0, 2 * KC)
    tl.store(Op + qi2[:, None] * (2 * KC) + d2[None, :], catd)

@triton.jit
def k_join(Xp, Op, BM: tl.constexpr, KC: tl.constexpr):
    # tl.join stacks on a NEW last axis: [BM,KC]+[BM,KC] -> [BM,KC,2]; reshape to [BM,2*KC] is interleaved,
    # so this is only a primitive-availability check, not the layout we want.
    qi = tl.arange(0, BM); dc = tl.arange(0, KC)
    a = tl.load(Xp + qi[:, None] * (2 * KC) + dc[None, :])
    b = tl.load(Xp + qi[:, None] * (2 * KC) + (KC + dc)[None, :])
    j = tl.join(a, b)                                  # [BM, KC, 2]
    r = tl.reshape(j, (BM, 2 * KC))
    qi2 = tl.arange(0, BM); d2 = tl.arange(0, 2 * KC)
    tl.store(Op + qi2[:, None] * (2 * KC) + d2[None, :], r)

@triton.jit
def k_listcarry(Xp, Op, BM: tl.constexpr, N: tl.constexpr, KC: tl.constexpr, NT: tl.constexpr):
    # list literal of N accumulators, carried across a REAL loop, indexed by constexpr, stored per-chunk
    qi = tl.arange(0, BM); dc = tl.arange(0, KC)
    accs = [tl.zeros((BM, KC), tl.float32) for _ in range(N)]   # list literal (comprehension form)
    for j in range(0, NT):
        x = tl.load(Xp + qi[:, None] * KC + dc[None, :]) + j
        accs = [accs[c] + x for c in range(N)]                  # rebind each iter
    for c in range(N):
        tl.store(Op + qi[:, None] * (N * KC) + (c * KC + dc)[None, :], accs[c])

@triton.jit
def k_tuplecarry(Xp, Op, BM: tl.constexpr, N: tl.constexpr, KC: tl.constexpr, NT: tl.constexpr):
    qi = tl.arange(0, BM); dc = tl.arange(0, KC)
    a0 = tl.zeros((BM, KC), tl.float32); a1 = tl.zeros((BM, KC), tl.float32)
    accs = (a0, a1)                                             # explicit tuple literal, N=2
    for j in range(0, NT):
        x = tl.load(Xp + qi[:, None] * KC + dc[None, :]) + j
        accs = (accs[0] + x, accs[1] + x)
    tl.store(Op + qi[:, None] * (2 * KC) + (0 * KC + dc)[None, :], accs[0])
    tl.store(Op + qi[:, None] * (2 * KC) + (1 * KC + dc)[None, :], accs[1])

def try_kernel(name, fn, shape_in, shape_out, **kw):
    dev = "cuda"
    X = torch.randn(*shape_in, device=dev, dtype=torch.float16)
    O = torch.zeros(*shape_out, device=dev, dtype=torch.float32)
    try:
        fn[(1,)](X, O, **kw)
        print(f"  {name}: OK  out.sum={O.sum().item():.3f}", flush=True)
    except Exception as e:
        msg = " | ".join(str(e).split('\n'))[:240]
        print(f"  {name}: FAIL  {type(e).__name__}: {msg}", flush=True)

if __name__ == "__main__":
    print(f"gfx: {torch.cuda.get_device_name(0)} triton {triton.__version__}", flush=True)
    try_kernel("A gen-tuple-index", k_genidx, (BM, 8 * KC), (BM, KC), BM=BM, N=8, KC=KC)
    try_kernel("B tl.cat last-axis", k_cat, (BM, 2 * KC), (BM, 2 * KC), BM=BM, KC=KC)
    try_kernel("C tl.join+reshape", k_join, (BM, 2 * KC), (BM, 2 * KC), BM=BM, KC=KC)
    try_kernel("D list-literal carry", k_listcarry, (BM, KC), (BM, 8 * KC), BM=BM, N=8, KC=KC, NT=4)
    try_kernel("E tuple-literal carry", k_tuplecarry, (BM, KC), (BM, 2 * KC), BM=BM, N=2, KC=KC, NT=4)
