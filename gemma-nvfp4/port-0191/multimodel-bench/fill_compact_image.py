#!/usr/bin/env python3
"""Timed bench legs for the multi-model matrix. Always targets model alias 'qwen' on :8011
(every model is served under that alias), so the same harness compares all three.
Modes:
  fill     <depth_k>   : single cold MAX-context prefill (unique nonce -> cache miss), max_tokens=8,
                         wall ~= raw prefill time. Prints {prompt_tokens, wall_s}.
  compact  <depth_k>   : compaction round-trip — long context -> summarize (generation) -> 1 continue turn.
                         Prints {ctx_tokens, summary_wall_s, continue_wall_s, total_wall_s}.
  image    [path]      : vision capability — send an image + 'describe', max_tokens=80. Prints
                         {supported, wall_s, reply}. supported=false if the server rejects images.
  audio    [path]      : audio capability probe; skips with a note unless an asset+support exist.
Output: one JSON object per run on stdout (prefixed RESULT: ), plus human lines.
"""
import sys, json, time, base64, urllib.request, urllib.error, os

URL = "http://localhost:8011/v1/chat/completions"
MODEL = "qwen"  # alias under which every model is served
NONCE = os.environ.get("BENCH_NONCE", "n0")
FILLER = ("Synthetic test document with numbered sections of generic filler text; "
    "each line describes fictitious operational parameters, counts and identifiers from an example "
    "system, with no relation to real data. ")

def post(body, timeout):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        return None, time.time()-t0, f"HTTP {e.code}: {e.read()[:200].decode('utf8','ignore')}"
    except Exception as e:
        return None, time.time()-t0, f"{type(e).__name__}: {str(e)[:200]}"
    return r, time.time()-t0, None

def big_prompt(depth_k, nonce):
    target_chars = int(depth_k*1000*3.6)
    parts=[f"EXTENDED OPERATIONAL REPORT nonce={nonce}.\n"]; n=0
    while sum(len(p) for p in parts) < target_chars:
        parts.append(f"[section {n:05d} n{nonce}] " + FILLER); n+=1
    return "".join(parts)

def emit(d):
    print("RESULT: " + json.dumps(d)); sys.stdout.flush()

def main():
    mode = sys.argv[1]
    if mode == "fill":
        depth = int(sys.argv[2]) if len(sys.argv)>2 else 250
        p = big_prompt(depth, NONCE+"_fill")
        r,wall,err = post({"model":MODEL,"messages":[{"role":"user","content":p}],"max_tokens":8,
                           "temperature":0.1,"chat_template_kwargs":{"enable_thinking":False}}, 700)
        if err: emit({"mode":"fill","ok":False,"err":err,"wall_s":round(wall,1)}); return
        emit({"mode":"fill","ok":True,"prompt_tokens":r.get("usage",{}).get("prompt_tokens"),
              "wall_s":round(wall,1)})
    elif mode == "compact":
        depth = int(sys.argv[2]) if len(sys.argv)>2 else 180
        ctx = big_prompt(depth, NONCE+"_compact")
        # 1) compaction: summarize the long context
        sumprompt = ctx + ("\n\nTASK: Summarize the ENTIRE document above in a paragraph of ~150 words "
                           "that preserves the key operational data. Return only the summary.")
        r1,w1,err = post({"model":MODEL,"messages":[{"role":"user","content":sumprompt}],"max_tokens":400,
                          "temperature":0.1,"chat_template_kwargs":{"enable_thinking":False}}, 700)
        if err: emit({"mode":"compact","ok":False,"err":err,"summary_wall_s":round(w1,1)}); return
        summary = r1.get("choices",[{}])[0].get("message",{}).get("content","") or ""
        ctx_tok = r1.get("usage",{}).get("prompt_tokens")
        # 2) continue: a short follow-up turn on the compacted summary
        r2,w2,err2 = post({"model":MODEL,"messages":[
                {"role":"user","content":"Context summary:\n"+summary},
                {"role":"assistant","content":"Understood, I have the summary."},
                {"role":"user","content":"Based on the summary, name two operational priorities."}],
                "max_tokens":120,"temperature":0.1,"chat_template_kwargs":{"enable_thinking":False}}, 200)
        emit({"mode":"compact","ok":True,"ctx_tokens":ctx_tok,"summary_wall_s":round(w1,1),
              "continue_wall_s":round(w2,1),"total_wall_s":round(w1+w2,1),
              "summary_chars":len(summary)})
    elif mode == "image":
        # default asset is bundled next to this script; override via argv[2] or BENCH_IMAGE_PATH
        path = sys.argv[2] if len(sys.argv)>2 else \
            os.environ.get("BENCH_IMAGE_PATH",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "image1.png"))
        b64 = base64.b64encode(open(path,"rb").read()).decode()
        durl = f"data:image/png;base64,{b64}"
        r,wall,err = post({"model":MODEL,"messages":[{"role":"user","content":[
                {"type":"text","text":"Describe this image in a single sentence."},
                {"type":"image_url","image_url":{"url":durl}}]}],
                "max_tokens":200,"temperature":0.2,
                "chat_template_kwargs":{"enable_thinking":False}}, 120)
        if err:
            # HTTP error usually means the model rejects image input -> genuinely not vision-capable
            emit({"mode":"image","supported":False,"reason":"rejected","err":err,"wall_s":round(wall,1)}); return
        msg = r.get("choices",[{}])[0].get("message",{}) or {}
        reply = (msg.get("content") or "")
        reasoning = (msg.get("reasoning") or "")
        emit({"mode":"image","supported":bool(reply.strip()),
              "reason":("ok" if reply.strip() else "empty_content"),
              "wall_s":round(wall,1),"reply":reply[:200],"reasoning_len":len(reasoning)})
    elif mode == "audio":
        emit({"mode":"audio","supported":False,"note":"Gemma-4 / Qwen3.6 are text+vision families; "
              "no audio input modality and no audio asset — skipped."})
    else:
        print("unknown mode", mode); sys.exit(2)

main()
