#!/usr/bin/env python3
"""Cold-vs-warm determinism probe (prefix-cache correctness).

At temperature 0, a replayed identical prompt must produce identical text.
Divergence between a cold run and a warm (cache-hit) replay indicates cached
recurrent-state corruption (relevant for hybrid GDN/mamba models with aligned
prefix caching). Note: divergence between CONCURRENT batched requests is
expected batch non-invariance, not cache corruption — always run a
unique-prefix concurrent control before concluding anything.

Usage: determinism-probe.py [--url http://localhost:8000] [--prompt-tokens 6000]
"""
import argparse, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--prompt-tokens", type=int, default=6000)
ap.add_argument("--salts", type=int, default=2)
ap.add_argument("--max-tokens", type=int, default=128)
args = ap.parse_args()

FILL = ("Context: regional sales, inventory coverage, receivables aging, "
        "demand forecasts, churn signals. ")
MODEL = json.loads(urllib.request.urlopen(args.url + "/v1/models", timeout=10)
                   .read())["data"][0]["id"]

def one(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": args.max_tokens,
                       "ignore_eos": True, "temperature": 0}).encode()
    req = urllib.request.Request(args.url + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return d["choices"][0]["text"], time.time() - t0

bad = False
for s in range(args.salts):
    filler = f"probe{s} " + FILL
    p = (filler * (args.prompt_tokens * 4 // len(filler) + 1))[: args.prompt_tokens * 4] \
        + "\n[request] Summarize the context above:"
    cold, tc = one(p)
    w1, t1 = one(p)
    w2, t2 = one(p)
    ok = cold == w1 == w2
    bad |= not ok
    print(f"salt {s}: cold {tc:5.2f}s | warm {'==' if ok else '!='} cold ({t1:.2f}s/{t2:.2f}s)")
print("PASS: cold == warm on all salts" if not bad else "FAIL ⚠ cache-path divergence")
