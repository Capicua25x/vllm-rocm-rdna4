#!/usr/bin/env python3
"""Rank GPU-kernel self-time from a torch/kineto chrome trace; split attention vs MoE.
Usage: parse_trace.py prof/<file>.pt.trace.json[.gz]   (defaults to newest in prof/)
"""
import sys, gzip, json, glob, os, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    cands = sorted(glob.glob(os.path.join(HERE, "prof", "*.json.gz")) + glob.glob(os.path.join(HERE, "prof", "*.json")),
                   key=os.path.getmtime)
    if not cands:
        print("no trace files in prof/"); sys.exit(1)
    path = cands[-1]
print(f"trace: {path}")
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    data = json.load(f)
events = data.get("traceEvents", data if isinstance(data, list) else [])

# GPU kernel events: ph=="X" and cat indicates a device kernel. ROCm/kineto uses cat "kernel".
def is_kernel(e):
    if e.get("ph") != "X":
        return False
    cat = str(e.get("cat", "")).lower()
    return cat in ("kernel", "gpu_op", "gpu_memcpy", "gpu_memset") or "kernel" in cat

kern_time = defaultdict(float)
kern_cnt = defaultdict(int)
flops_by_op = defaultdict(float)
total_kern = 0.0
for e in events:
    if is_kernel(e):
        nm = e.get("name", "?")
        d = float(e.get("dur", 0) or 0)
        kern_time[nm] += d
        kern_cnt[nm] += 1
        total_kern += d
    args = e.get("args") or {}
    fl = args.get("flops") or args.get("Flops") or 0
    if fl:
        flops_by_op[e.get("name", "?")] += float(fl)

def cat_of(name):
    n = name.lower()
    if any(k in n for k in ("attention", "unified_attn", "flash", "_attn", "paged", "softmax")):
        return "ATTENTION"
    if any(k in n for k in ("moe", "fp4", "fused_experts", "grouped_gemm", "expert", "topk", "silu_mul", "act_and_mul")):
        return "MoE"
    if any(k in n for k in ("gemm", "matmul", "linear", "mm_", "_mm", "cutlass", "wmma", "dot")):
        return "GEMM/proj"
    if any(k in n for k in ("rms", "norm", "rope", "rotary", "embed", "cast", "copy", "elementwise", "add", "mul")):
        return "norm/rope/elt"
    return "other"

cat_time = defaultdict(float)
for nm, t in kern_time.items():
    cat_time[cat_of(nm)] += t

print(f"\n=== total GPU-kernel time captured: {total_kern/1e3:.1f} ms ===")
print("\n--- by CATEGORY (self-time) ---")
for c, t in sorted(cat_time.items(), key=lambda x: -x[1]):
    print(f"  {c:16} {t/1e3:9.2f} ms   {100*t/total_kern:5.1f}%")
print("\n--- top 25 kernels by self-time ---")
for nm, t in sorted(kern_time.items(), key=lambda x: -x[1])[:25]:
    print(f"  {t/1e3:9.2f} ms  {100*t/total_kern:5.1f}%  x{kern_cnt[nm]:<4}  [{cat_of(nm)}]  {nm[:90]}")
if flops_by_op:
    print("\n--- top 12 ops by FLOPs (with_flops=true) ---")
    for nm, fl in sorted(flops_by_op.items(), key=lambda x: -x[1])[:12]:
        print(f"  {fl/1e9:11.1f} GFLOP  [{cat_of(nm)}]  {nm[:80]}")
