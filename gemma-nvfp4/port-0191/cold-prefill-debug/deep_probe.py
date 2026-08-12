#!/usr/bin/env python3
"""Profile ONE prefill chunk AT DEPTH to attribute cost to attention-2D vs NVFP4-MoE kernels.

Warm a ~Dk prefix into the prefix cache, then fire (prefix + small cold suffix). That suffix
forward is a small-query x deep-KV attention call — structurally identical to a cold prefill
chunk at depth D — so the torch trace's per-kernel self-time is the at-depth attention-vs-MoE
split. Run AFTER the depth curve, with the profiler armed (PROFILER=1 launch).
Usage: deep_probe.py [depth=120000] [suffix_tokens=2000]
"""
import os, sys, time, json, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cold_prefill_test import build_prompt, FILLER  # noqa
URL = "http://localhost:8011/v1/chat/completions"

D = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
SUF = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SALT = f"PROBE-{D}-{time.time_ns()}"

def fire(content, mt=4, to=3000):
    body = {"model": "gemma", "messages": [{"role": "user", "content": content}],
            "max_tokens": mt, "temperature": 0.1, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=to).read())
    return time.time() - t0, r.get("usage", {})

base = f"{SALT} BASE.\n" + build_prompt(D)
print(f"[deep_probe] warming {D}-token prefix into cache ...", flush=True)
w, u = fire(base)
print(f"[deep_probe] warm done in {w:.1f}s prompt_tokens={u.get('prompt_tokens')}", flush=True)

# suffix shares the full base prefix (cache HIT) + ~SUF cold tokens at depth D
suffix = base + "\n\n[ANEXO DE PROFUNDIDAD] " + (FILLER * (SUF // 40 + 1))
print(f"[deep_probe] start_profile + deep suffix forward (cold {SUF}-tok chunk @ depth {D}) ...", flush=True)
import urllib.request as U
U.urlopen(U.Request("http://localhost:8011/start_profile", b""), timeout=10).read()
p, u2 = fire(suffix)
U.urlopen(U.Request("http://localhost:8011/stop_profile", b""), timeout=10).read()
print(f"[deep_probe] deep forward done in {p:.1f}s prompt_tokens={u2.get('prompt_tokens')} "
      f"cached_implied={u2.get('prompt_tokens',0)-(u.get('prompt_tokens',0))} new", flush=True)
print("[deep_probe] DONE — trace under cold-prefill-debug/prof/", flush=True)
