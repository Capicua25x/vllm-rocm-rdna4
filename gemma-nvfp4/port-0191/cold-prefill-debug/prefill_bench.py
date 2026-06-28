#!/usr/bin/env python3
"""Cold-prefill A/B timer. Fires single unique requests at fixed token depths,
measures wall time (~= prefill time, max_tokens tiny). Unique nonce per request
defeats prefix caching so every run is a true COLD prefill.
Usage: prefill_bench.py <tag> <depth_k> [depth_k ...]   e.g. prefill_bench.py CHUNK128 63 120
"""
import sys, time, json, urllib.request

URL = "http://localhost:8011/v1/chat/completions"; MODEL = "gemma"
FILLER = ("This synthetic test document contains numbered sections of generic filler "
    "text; each line describes fictitious operational parameters, counts, identifiers and "
    "observations from an example system, with no relation to real data. ")

def build(target_tokens, nonce):
    tc = int(target_tokens * 3.6)
    parts = [f"SYNTHETIC TEST DOCUMENT nonce={nonce}.\n"]; n = 0
    while sum(len(p) for p in parts) < tc:
        parts.append(f"[section {n:05d} n{nonce}] " + FILLER); n += 1
    parts.append("\n\nSummarize in 20 words.")
    return "".join(parts)

def fire(target_tokens, nonce):
    prompt = build(target_tokens, nonce)
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=2400).read())
    dt = time.time() - t0
    pt = resp.get("usage", {}).get("prompt_tokens", -1)
    return dt, pt

if __name__ == "__main__":
    tag = sys.argv[1]
    depths = [int(x) for x in sys.argv[2:]]
    print(f"=== {tag} ===", flush=True)
    for d in depths:
        for p in (1, 2):  # 2 passes for non-determinism
            nonce = f"{tag}_{d}_{p}"
            try:
                dt, pt = fire(d * 1000, nonce)
                print(f"  depth~{d}k  pass{p}  prompt_tokens={pt:>7}  wall={dt:8.1f}s", flush=True)
            except Exception as e:
                print(f"  depth~{d}k  pass{p}  FAIL: {type(e).__name__}: {str(e)[:120]}", flush=True)
