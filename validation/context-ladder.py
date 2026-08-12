#!/usr/bin/env python3
"""Long-context survival ladder ("compaction test").

Sends progressively larger single prompts (conversation-shaped filler + a
summarize instruction) and reports latency, prefill rate and a sample of the
output. Also probes the adversarial chunked-prefill edge: a prompt whose token
count leaves a tiny (<8-token) final scheduler chunk, plus a warm replay to
check determinism (temp 0).

Usage: context-ladder.py [--url http://localhost:8000] [--model auto]
                         [--sizes 32000,100000,200000] [--gen 128]
"""
import argparse, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--model", default="auto")
ap.add_argument("--sizes", default="32000,100000,200000")
ap.add_argument("--gen", type=int, default=128)
args = ap.parse_args()

FILLER = ("[user]: how are the northern region numbers trending this month? "
          "[assistant]: Northern region volume is up twelve percent month over "
          "month, led by the two top product lines; margins are holding steady. ")

def detect_model():
    d = json.loads(urllib.request.urlopen(args.url + "/v1/models", timeout=10).read())
    return d["data"][0]["id"]

MODEL = detect_model() if args.model == "auto" else args.model

def make_prompt(tokens, salt):
    filler = f"[{salt}] " + FILLER
    body = (filler * (tokens * 4 // len(filler) + 1))[: tokens * 4]
    return body + "\n[system]: Summarize the conversation above in one paragraph:"

def one(prompt, max_tokens):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "ignore_eos": True, "temperature": 0}).encode()
    req = urllib.request.Request(args.url + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    except Exception as e:
        return None, time.time() - t0, str(e)[:120]
    u = d.get("usage", {})
    return (u.get("prompt_tokens"), u.get("completion_tokens"),
            d["choices"][0]["text"]), time.time() - t0, None

print(f"{'target':>8} | {'prompt_tok':>10} | {'lat s':>7} | {'prefill tok/s':>13} | sample")
for i, sz in enumerate(int(x) for x in args.sizes.split(",")):
    r, dt, err = one(make_prompt(sz, f"rung{i}"), args.gen)
    if err:
        print(f"{sz:>8} | {'—':>10} | {dt:>7.1f} | ERROR: {err}")
        continue
    ptok, ctok, txt = r
    pf = ptok / max(dt - ctok / 100.0, 0.01)
    print(f"{sz:>8} | {ptok:>10} | {dt:>7.1f} | {pf:>13.0f} | {txt[:50]!r}")

# adversarial tiny final chunk (assumes a 16384-token scheduler budget)
p = make_prompt(90000, "adv")
r, _, err = one(p, 1)
if not err:
    ptok = r[0]
    target = (ptok // 16384) * 16384 + 4
    for _ in range(6):
        delta = ptok - target
        if abs(delta) <= 2: break
        p = p[: len(p) - delta * 4] if delta > 0 else p + " data" * (-delta)
        r, _, err = one(p, 1)
        if err: break
        ptok = r[0]
    a, dt1, _ = one(p, 64)
    b, dt2, _ = one(p, 64)   # warm replay
    if a and b:
        print(f"adversarial: prompt={a[0]} (mod 16384 = {a[0] % 16384}) · "
              f"cold==warm: {'YES' if a[2] == b[2] else 'NO ⚠'} ({dt1:.1f}s/{dt2:.1f}s)")
