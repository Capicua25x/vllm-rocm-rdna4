#!/usr/bin/env python3
"""
Cold large-prefill admission test for the Gemma-4 :8011 instance.

Reproduces the production "won't survive a compact" failure: a single FRESH
request whose prompt exceeds the GPU KV cache size (167,296 tokens) — i.e. a
compaction whose prefix got evicted, or any big cold paste — gets refused at
the scheduler admission gate (scheduler_reserve_full_isl=True -> can_fit_full_sequence
demands the full ISL fit in free blocks at once -> break -> Waiting forever, KV 0%).

BEFORE (reserve_full_isl=True):  expect HANG (no admission, times out).
AFTER  (--no-scheduler-reserve-full-isl): expect ADMIT -> prefill -> summary returns,
        and (with --concurrent) coexists with light concurrent traffic.

Usage:
  python cold_prefill_test.py --target-tokens 200000 --timeout 200
  python cold_prefill_test.py --target-tokens 200000 --timeout 300 --concurrent 3
"""
import argparse, json, sys, time, threading, urllib.request, urllib.error

URL = "http://localhost:8011/v1/chat/completions"
MODEL = "gemma"

# 20 needle canaries scattered through the long context. Retrieval after a long
# cold prefill verifies the run is COHERENT, not merely non-hanging.
CANARIES = [
    # Synthetic, domain-neutral needle facts — only the embedded values matter for recall.
    ("FACT-01", "the-sensor-code-is-ZX4417"),
    ("FACT-02", "the-north-node-logged-8821-events"),
    ("FACT-03", "the-group-identifier-is-G-Polaris"),
    ("FACT-04", "the-threshold-rose-to-1375-units"),
    ("FACT-05", "the-flagged-batch-is-LT-99213"),
    ("FACT-06", "the-cycle-goal-is-44000-items"),
    ("FACT-07", "the-assigned-operator-is-Alex-Morgan"),
    ("FACT-08", "the-assigned-unit-is-code-T-50182"),
    ("FACT-09", "the-agreed-adjustment-is-7-percent"),
    ("FACT-10", "the-main-route-runs-through-the-Blue-Sector"),
    ("FACT-11", "the-pending-balance-is-318450-units"),
    ("FACT-12", "the-highlighted-element-is-module-type-I"),
    ("FACT-13", "the-record-total-is-1284900"),
    ("FACT-14", "the-secondary-depot-is-in-the-West-Zone"),
    ("FACT-15", "the-defined-deadline-is-48-hours"),
    ("FACT-16", "the-technical-contact-is-Robin-Cole"),
    ("FACT-17", "the-record-number-is-CD-77410"),
    ("FACT-18", "the-measured-rate-is-2.3-percent"),
    ("FACT-19", "the-deadline-falls-on-the-15th"),
    ("FACT-20", "the-backup-is-model-GR-880"),
]

FILLER = (
    "This synthetic test document contains numbered sections of generic filler "
    "text; each line describes fictitious operational parameters, counts, "
    "identifiers and observations from an example system, with no relation to real "
    "data. "
)

def build_prompt(target_tokens: int) -> str:
    # ~3.6 chars/token for this English text; build by chars then trust server count.
    target_chars = int(target_tokens * 3.6)
    parts = ["SYNTHETIC TEST DOCUMENT. Read the whole document.\n\n"]
    n = 0
    ci = 0
    while sum(len(p) for p in parts) < target_chars:
        parts.append(f"[section {n:05d}] " + FILLER)
        # sprinkle a canary every ~ (lines/20) sections, spread across full depth
        if n > 0 and ci < len(CANARIES) and n % 90 == 0:
            tag, val = CANARIES[ci]
            parts.append(f"\n>>> IMPORTANT NOTE {tag}: {val} <<<\n")
            ci += 1
        n += 1
    # any canaries not yet placed (short doc): append before the question
    while ci < len(CANARIES):
        tag, val = CANARIES[ci]; parts.append(f"\n>>> IMPORTANT NOTE {tag}: {val} <<<\n"); ci += 1
    parts.append(
        "\n\nEND OF DOCUMENT.\n\nTASK: Summarize the report in 120 words and "
        "also answer exactly: what is the value of FACT-07 and FACT-13?"
    )
    return "".join(parts)

def post(body: dict, timeout: float):
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return time.time() - t0, json.loads(r.read().decode()), None
    except urllib.error.URLError as e:
        return time.time() - t0, None, f"{type(e).__name__}: {e}"
    except Exception as e:
        return time.time() - t0, None, f"{type(e).__name__}: {e}"

def concurrent_chatter(stop_evt: threading.Event, results: list, idx: int):
    """Simulate 'a few people using the server' — small chats every couple seconds."""
    i = 0
    while not stop_evt.is_set():
        dt, resp, err = post({
            "model": MODEL,
            "messages": [{"role": "user", "content": f"In one sentence, tell me a fun fact number {idx}-{i}."}],
            "max_tokens": 60, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=60)
        ok = resp is not None and not err
        results.append((ok, round(dt, 1)))
        i += 1
        stop_evt.wait(2.0)

def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=200000)
    ap.add_argument("--timeout", type=float, default=200)
    ap.add_argument("--concurrent", type=int, default=0, help="N background small-chat threads")
    ap.add_argument("--think", action="store_true", help="enable_thinking=True (closer to real compaction)")
    ap.add_argument("--model", default=MODEL, help="served-model-name to hit (gemma | qwen). Default gemma.")
    args = ap.parse_args()
    MODEL = args.model

    prompt = build_prompt(args.target_tokens)
    print(f"[prompt] chars={len(prompt):,}  (~est {len(prompt)//4:,}-{len(prompt)//3:,} tokens)", flush=True)

    stop_evt = threading.Event()
    chat_results: list = []
    threads = []
    if args.concurrent > 0:
        for k in range(args.concurrent):
            t = threading.Thread(target=concurrent_chatter, args=(stop_evt, chat_results, k), daemon=True)
            t.start(); threads.append(t)
        print(f"[concurrent] started {args.concurrent} background chatters", flush=True)
        time.sleep(3)  # let some small reqs occupy blocks first

    big_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300, "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": bool(args.think)},
    }
    print(f"[big] firing cold {args.target_tokens//1000}k-token request, timeout={args.timeout}s, think={args.think} ...", flush=True)
    dt, resp, err = post(big_body, timeout=args.timeout)
    stop_evt.set()

    print("\n" + "=" * 60)
    if err or resp is None:
        print(f"[big] RESULT: ❌ NO RESPONSE in {dt:.0f}s  ({err})")
        print("[big] => consistent with admission HANG (request never scheduled).")
        verdict = "HANG"
    else:
        usage = resp.get("usage", {})
        msg = resp["choices"][0]["message"]
        content = (msg.get("content") or "")
        pt = usage.get("prompt_tokens")
        print(f"[big] RESULT: ✅ COMPLETED in {dt:.0f}s  prompt_tokens={pt}  completion_tokens={usage.get('completion_tokens')}")
        d07 = "GR" not in content and ("alex" in content.lower() or "morgan" in content.lower())
        d13 = "1284900" in content.replace(",", "").replace(".", "") or "1,284,900" in content
        print(f"[big] canary FACT-07 (Alex Morgan) present: {d07}")
        print(f"[big] canary FACT-13 (1,284,900) present:    {d13}")
        print(f"[big] answer snippet: {content[:400]!r}")
        verdict = "COMPLETED"
    if args.concurrent > 0:
        ok = sum(1 for r in chat_results if r[0]); tot = len(chat_results)
        lat = [r[1] for r in chat_results if r[0]]
        avg = sum(lat) / len(lat) if lat else 0
        print(f"[concurrent] small chats during big req: {ok}/{tot} ok, avg {avg:.1f}s "
              f"(proves coexistence; small reqs not starved by the big prefill)")
    print("=" * 60)
    print(f"VERDICT: {verdict}")

if __name__ == "__main__":
    main()
