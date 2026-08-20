"""CPU unit test of the stop-strings-in-reasoning guard (no GPU, no server).

Needs any tokenizer with a <think> token; override with GUARD_TOKENIZER.
Run: python3 tests/v1/engine/test_suppress_stops_in_reasoning.py
"""
import os
import importlib.util, sys
from types import SimpleNamespace

DET = os.environ.get("GUARD_DET_PATH",
                     os.path.join(os.path.dirname(__file__), "../../../vllm/v1/engine/detokenizer.py"))
spec = importlib.util.spec_from_file_location("det_patched", DET)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ.get("GUARD_TOKENIZER", "Qwen/Qwen3-4B"))
think_id = tok.convert_tokens_to_ids("<think>")
assert isinstance(think_id, int) and think_id >= 0, f"no <think> token: {think_id}"

def req(prompt_ids, stop):
    sp = SimpleNamespace(stop=stop, min_tokens=0, include_stop_str_in_output=False,
                         skip_special_tokens=False, spaces_between_special_tokens=True,
                         output_kind=None, logprobs=None)
    return SimpleNamespace(sampling_params=sp, prompt_token_ids=prompt_ids,
                           request_id="t1")

def ids(s): return tok.encode(s, add_special_tokens=False)

# A) thinking request: stop must stay dormant inside reasoning, fire after </think>
d = m.IncrementalDetokenizer.from_new_request(tok, req([1, 2, think_id], ["Question:"]))
assert getattr(d, "_reasoning_stop_guard", False), "guard did not arm"
r1 = d.update(ids("I restate the Question: several times while thinking. "), False)
assert r1 is None, f"stop fired inside reasoning: {r1!r}"
r2 = d.update(ids("done </think> The answer is 42. "), False)
assert r2 is None, f"stop fired without stop text in content: {r2!r}"
r3 = d.update(ids("Question: should stop here"), False)
assert r3 == "Question:", f"stop did not fire in content: {r3!r}"
print("A ok: dormant in reasoning, fires in content")

# B) non-thinking request: stop fires normally
d = m.IncrementalDetokenizer.from_new_request(tok, req([1, 2, 3], ["Question:"]))
assert not getattr(d, "_reasoning_stop_guard", False)
r = d.update(ids("Here is a Question: to stop on"), False)
assert r == "Question:", f"non-thinking stop broken: {r!r}"
print("B ok: non-thinking unaffected")

# C) one-chunk case: same update carries trailing reasoning (with stop text) + </think>
d = m.IncrementalDetokenizer.from_new_request(tok, req([1, think_id], ["Question:"]))
r = d.update(ids("thinking about the Question: again </think> clean content"), False)
assert r is None, f"stop fired from reasoning inside the closing chunk: {r!r}"
r = d.update(ids(" now Question: fires"), False)
assert r == "Question:", f"post-close stop broken: {r!r}"
print("C ok: one-chunk leak guarded")

# D) opt-out env
os.environ["VLLM_SUPPRESS_STOPS_IN_REASONING"] = "0"
d = m.IncrementalDetokenizer.from_new_request(tok, req([1, think_id], ["Question:"]))
assert not getattr(d, "_reasoning_stop_guard", False), "opt-out ignored"
print("D ok: opt-out honored")
print("ALL GUARD TESTS PASS")
