#!/usr/bin/env python3
"""Near-256k multi-needle retrieval head-to-head: Gemma vs Qwen on :8011.
Embeds 20 canaries spread across the full depth, asks for ALL of them, scores recall.
Usage: needle256k.py <model> <target_tokens> [timeout]
"""
import sys, json, time, urllib.request, unicodedata

URL = "http://localhost:8011/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 315000
TIMEOUT = float(sys.argv[3]) if len(sys.argv) > 3 else 700.0

# Domain-neutral synthetic filler — only the token count matters for the recall test.
FILLER = ("This synthetic test document contains numbered sections of generic filler "
    "text; each line describes fictitious operational parameters, counts, identifiers and "
    "observations from an example system, with no relation to real data. ")

# (tag, embedded value, [acceptable answer needles — normalized lowercase no-accent])
# Synthetic, business-neutral. FACT-09/10/12/14/15/19 keep space-containing answer needles on
# purpose (the embedded values are hyphenated) so the norm() hyphen-neutralization stays exercised.
CANARIES = [
 ("FACT-01","the-sensor-code-is-ZX4417",["zx4417"]),
 ("FACT-02","the-north-node-logged-8821-events",["8821"]),
 ("FACT-03","the-group-identifier-is-G-Polaris",["polaris"]),
 ("FACT-04","the-threshold-rose-to-1375-units",["1375"]),
 ("FACT-05","the-flagged-batch-is-LT-99213",["99213"]),
 ("FACT-06","the-cycle-goal-is-44000-items",["44000"]),
 ("FACT-07","the-assigned-operator-is-Alex-Morgan",["alex","morgan"]),
 ("FACT-08","the-assigned-unit-is-code-T-50182",["50182"]),
 ("FACT-09","the-agreed-adjustment-is-7-percent",["7 percent","7%","7 %","seven percent"]),
 ("FACT-10","the-main-route-runs-through-the-Blue-Sector",["blue sector"]),
 ("FACT-11","the-pending-balance-is-318450-units",["318450"]),
 ("FACT-12","the-highlighted-element-is-module-type-I",["module type i"]),
 ("FACT-13","the-record-total-is-1284900",["1284900"]),
 ("FACT-14","the-secondary-depot-is-in-the-West-Zone",["west zone"]),
 ("FACT-15","the-defined-deadline-is-48-hours",["48 hour"]),
 ("FACT-16","the-technical-contact-is-Robin-Cole",["robin","cole"]),
 ("FACT-17","the-record-number-is-CD-77410",["77410"]),
 ("FACT-18","the-measured-rate-is-2.3-percent",["2.3","2,3"]),
 ("FACT-19","the-deadline-falls-on-the-15th",["15th","day 15","the 15"]),
 ("FACT-20","the-backup-is-model-GR-880",["gr-880","gr880","gr 880"]),
]

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode().lower()
    # drop thousands separators (1,284,900==1284900) AND neutralize hyphens: the embedded values are
    # hyphenated ("7-percent") and the model echoes them verbatim ("exact value"), but the answer
    # needles use spaces ("7 percent") — without this, correct recalls of FACT-09/12/14/15/19 scored
    # as false misses (verified 2026-06-27: model output 20/20, scorer reported 15/20).
    s = s.replace(",", "").replace("-", " ")
    return " ".join(s.split())  # collapse whitespace

def build(target):
    target_chars = int(target * 3.6)
    parts = ["SYNTHETIC TEST DOCUMENT. Read the WHOLE document; it contains 20 IMPORTANT NOTES.\n\n"]
    n=ci=0
    step = 90
    while sum(len(p) for p in parts) < target_chars:
        parts.append(f"[section {n:05d}] " + FILLER)
        if n>0 and ci<len(CANARIES) and n % step == 0:
            tag,val,_ = CANARIES[ci]; parts.append(f"\n>>> IMPORTANT NOTE {tag}: {val} <<<\n"); ci+=1
        n+=1
    while ci<len(CANARIES):
        tag,val,_=CANARIES[ci]; parts.append(f"\n>>> IMPORTANT NOTE {tag}: {val} <<<\n"); ci+=1
    parts.append("\n\nEND OF DOCUMENT.\n\nTASK: The document contains 20 notes 'IMPORTANT NOTE FACT-XX'. "
        "For EACH one, from FACT-01 to FACT-20, write on one line: 'FACT-XX: <exact value>'. "
        "List all 20. Do not summarize, just list the exact values you found.")
    return "".join(parts)

prompt = build(TARGET)
print(f"[{MODEL}] prompt chars={len(prompt):,}", flush=True)
body = {"model": MODEL, "messages":[{"role":"user","content":prompt}],
        "max_tokens": 2200, "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False}}
t0=time.time()
req=urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type":"application/json"})
try:
    r=json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
except Exception as e:
    print(f"[{MODEL}] ERROR {type(e).__name__}: {e}"); sys.exit(1)
wall=time.time()-t0
ans=r.get("choices",[{}])[0].get("message",{}).get("content","") or ""
usage=r.get("usage",{})
na=norm(ans)
hits=[]
for tag,val,needles in CANARIES:
    ok=any(norm(nd) in na for nd in needles)
    hits.append((tag,ok))
score=sum(1 for _,ok in hits if ok)
print(f"[{MODEL}] prompt_tokens={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} wall={wall:.0f}s")
print(f"[{MODEL}] RECALL SCORE: {score}/20")
print(f"[{MODEL}] misses: {[t for t,ok in hits if not ok]}")
print(f"[{MODEL}] answer (first 900 chars):\n{ans[:900]}")
print("="*60)
