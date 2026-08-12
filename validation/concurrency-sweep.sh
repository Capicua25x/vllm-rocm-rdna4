#!/bin/bash
# Concurrency sweep: per-user and aggregate tok/s at several client counts.
# A shared (prefix-cacheable) prompt of --prompt-tokens N models a long system
# prompt; each request appends a unique tail. 0 = short prompt (compute-bound).
# Usage: ./concurrency-sweep.sh [--url http://localhost:8000] [--prompt-tokens 6000] [--levels "1 8 16 32"]
URL="http://localhost:8000"; PROMPT_TOKENS=0; LEVELS="1 8 16 32"; MAX_TOKENS=256
while [[ $# -gt 0 ]]; do case $1 in
  --url) URL="$2"; shift 2;; --prompt-tokens) PROMPT_TOKENS="$2"; shift 2;;
  --levels) LEVELS="$2"; shift 2;; --max-tokens) MAX_TOKENS="$2"; shift 2;;
  *) shift;; esac; done
MODEL=$(curl -s "$URL/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
URL="$URL" MODEL="$MODEL" MAX_TOKENS="$MAX_TOKENS" PROMPT_TOKENS="$PROMPT_TOKENS" LEVELS="$LEVELS" python3 - <<'PY'
import os, json, time, urllib.request, concurrent.futures
URL=os.environ["URL"]; MODEL=os.environ["MODEL"]
MAXTOK=int(os.environ["MAX_TOKENS"]); PT=int(os.environ["PROMPT_TOKENS"])
BASE="Write a long, detailed essay about logistics planning for a mid-size manufacturer:"
FILL="Context: regional sales, inventory coverage, receivables aging, demand forecasts, churn signals. "
PREFIX=(FILL*(PT*4//len(FILL)+1))[:PT*4] if PT>0 else ""
def one(uid):
    p=(PREFIX+f"\n[request {uid}] "+BASE) if PT>0 else BASE
    body=json.dumps({"model":MODEL,"prompt":p,"max_tokens":MAXTOK,"ignore_eos":True,"temperature":0}).encode()
    r=urllib.request.Request(URL+"/v1/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.time(); d=json.loads(urllib.request.urlopen(r,timeout=600).read()); dt=time.time()-t0
    return d.get("usage",{}).get("completion_tokens",MAXTOK), dt
one(0); one(1)  # warm the shared prefix
print(f"{'users':>5} | {'per-user tok/s':>14} | {'aggregate':>9}")
for n in [int(x) for x in os.environ["LEVELS"].split()]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        w0=time.time(); res=list(ex.map(one,range(n))); wall=time.time()-w0
    per=sum(c/t for c,t in res)/len(res); agg=sum(c for c,_ in res)/wall
    print(f"{n:>5} | {per:>14.1f} | {agg:>9.0f}")
PY
