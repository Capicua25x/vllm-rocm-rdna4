#!/usr/bin/env python3
"""Cold-prefill DEPTH CURVE via server-side TTFT (= prefill time, decode-independent).

prefill_tok/s(depth) curve is the decisive H1-vs-H2 discriminator:
  ∝ 1/depth  ⇒ attention-bound (the only O(n^2) term)   [H1]
  ~ flat     ⇒ MoE / linear-constant bound               [H2]
Each request is a FRESH (prefix-cache MISS) cold prefill. max_tokens tiny so wall≈prefill.
"""
import os, sys, time, json, urllib.request, urllib.error
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cold_prefill_test import build_prompt  # noqa

URL = "http://localhost:8011/v1/chat/completions"
MET = "http://localhost:8011/metrics"
DEPTHS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["40000","80000","120000","160000"])]

def cold(prompt, D):
    # Unique nonce in block 0 => different block-0 hash => cascades => NO prefix-cache hit.
    # Guarantees a full cold prefill even with --enable-prefix-caching on.
    return f"RUN-{D}-{time.time_ns()} EJECUCION UNICA NO CACHEAR.\n" + prompt

def metrics():
    t = urllib.request.urlopen(MET, timeout=8).read().decode()
    sm = cnt = pt = cached = 0.0
    for ln in t.splitlines():
        if ln.startswith("vllm:time_to_first_token_seconds_sum"): sm = float(ln.split()[-1])
        elif ln.startswith("vllm:time_to_first_token_seconds_count"): cnt = float(ln.split()[-1])
        elif ln.startswith("vllm:prompt_tokens_total"): pt = float(ln.split()[-1])
        elif ln.startswith("vllm:prompt_tokens_cached_total"): cached = float(ln.split()[-1])
    return sm, cnt, pt, cached

rows = []
for D in DEPTHS:
    prompt = cold(build_prompt(D), D)
    s0, c0, p0, k0 = metrics()
    body = {"model": "gemma", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4, "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    try:
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=3000).read())
        wall = time.time() - t0
        s1, c1, p1, k1 = metrics()
        dcount = max(1.0, c1 - c0)
        ttft = (s1 - s0) / dcount
        realpt = r.get("usage", {}).get("prompt_tokens") or int(p1 - p0)
        cachedhit = int(k1 - k0)
        tps = realpt / ttft if ttft > 0 else 0
        cold_ok = "COLD" if cachedhit < 0.05 * realpt else f"⚠CACHED({cachedhit})"
        row = f"depth~{D:>7}: prompt_tokens={realpt:>7}  cached={cachedhit:>6} [{cold_ok}]  TTFT(prefill)={ttft:7.1f}s  prefill_tok/s={tps:7.0f}  wall={wall:6.1f}s"
    except Exception as e:
        row = f"depth~{D:>7}: ERROR {type(e).__name__}: {e}"
    rows.append(row)
    print(row, flush=True)

print("\n==== DEPTH CURVE (prefill_tok/s vs depth) ====", flush=True)
for r in rows: print(r, flush=True)
print("Interpretation: tok/s halving as depth doubles ⇒ H1 attention-bound; flat ⇒ H2 MoE-bound.", flush=True)
