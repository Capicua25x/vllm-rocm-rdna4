#!/usr/bin/env python3
"""Binary-search Gemma's admission-deadlock cliff via num_requests_running.
ADMIT = running goes >=1 within obs window; DEADLOCK = stays 0. ~80s/probe, same server."""
import threading, time, json, urllib.request

URL="http://localhost:8011/v1/chat/completions"; MET="http://localhost:8011/metrics"; MODEL="gemma"
FILLER=("This synthetic test document contains numbered sections of generic filler "
    "text; each line describes fictitious operational parameters, counts, identifiers and "
    "observations from an example system, with no relation to real data. ")
RATIO=0.82  # actual tokens / target (empirical from Qwen 257925/315000)

def build(target):
    tc=int(target*3.6); parts=["SYNTHETIC TEST DOCUMENT.\n"]; n=0
    while sum(len(p) for p in parts)<tc:
        parts.append(f"[section {n:05d}] "+FILLER); n+=1
    parts.append("\n\nSummarize in 30 words."); return "".join(parts)

def metric(name):
    try:
        for ln in urllib.request.urlopen(MET,timeout=5).read().decode().splitlines():
            if ln.startswith(name): return float(ln.split()[-1])
    except Exception: pass
    return -1.0

def wait_free(maxs=120):
    for _ in range(maxs):
        if metric("vllm:num_requests_running")==0 and metric("vllm:num_requests_waiting")==0: return True
        time.sleep(1)
    return False

def probe(target, send_to=85, obs=78):
    prompt=build(target); h={}
    def send():
        body={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":6,
              "temperature":0.1,"chat_template_kwargs":{"enable_thinking":False}}
        try:
            r=urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"})
            urllib.request.urlopen(r,timeout=send_to).read(); h["ok"]=1
        except Exception as e: h["err"]=type(e).__name__
    th=threading.Thread(target=send,daemon=True); th.start()
    admitted=False; mx=0.0; t0=time.time()
    while time.time()-t0 < obs:
        r=metric("vllm:num_requests_running"); mx=max(mx,r)
        if r>=1: admitted=True; break
        time.sleep(4)
    act=int(target*RATIO)
    print(f"  target={target:>6} (~{act//1000}k actual): {'ADMIT' if admitted else 'DEADLOCK':>8}  (max_running={mx})",flush=True)
    th.join(timeout=send_to+5)  # let the request finish/abort
    if not wait_free(120): print("    WARN: server not free after probe", flush=True)
    time.sleep(3)
    return admitted

print(f"server free before start: {wait_free(30)}",flush=True)
lo, hi = 207000, 315000   # lo known ADMIT (~170k), hi known DEADLOCK (~253k)
print(f"binary search target [{lo} (~{int(lo*RATIO)//1000}k), {hi} (~{int(hi*RATIO)//1000}k)]",flush=True)
while hi-lo > 16000:
    mid=(lo+hi)//2
    if probe(mid): lo=mid
    else: hi=mid
print(f"\nCLIFF: ADMITS up to ~{int(lo*RATIO)//1000}k actual (target {lo}); DEADLOCKS at ~{int(hi*RATIO)//1000}k (target {hi})",flush=True)
