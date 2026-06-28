# Patch: `vllm/config/speculative.py` (vLLM 0.19.1 tree)

Role: config dispatch — register `gemma4_mtp` as an MTP model type, rewrite the
draft checkpoint's `gemma4_assistant` model_type to `gemma4_mtp` during
`hf_config_override`, and add the `use_gemma4_mtp()` predicate.

Target file (apply against):
`<vllm-src>/vllm/config/speculative.py`

Reference (0.23.0 upstream, already-correct shapes):
`vllm-0.23.0-vanilla/vllm/config/speculative.py`
 - rewrite block: :512-521
 - predicate: :1054-1060

## Verification of the nested-config fact (load-bearing)

The draft checkpoint config is NESTED. Confirmed from
`models--google--gemma-4-26B-A4B-it-assistant/.../config.json`:

    top-level  model_type            = "gemma4_assistant"
    top-level  num_kv_shared_layers  = (absent / None)
    text_config.model_type           = "gemma4_text"
    text_config.num_kv_shared_layers = 4        <-- the real field
    text_config.num_hidden_layers    = 4

Therefore the `num_kv_shared_layers = 0` reset MUST target `text_config`, not the
top-level config. Writing it at top-level would leave `text_config.num_kv_shared_layers=4`
intact and the draft model would try to self-share all 4 layers' KV instead of
sharing from the target (the Q-only draft has no k_proj/v_proj — self-sharing is
wrong and cross-model sharing is set up later by the proposer).

---

## HUNK 1 — register `gemma4_mtp` in `MTPModelTypes` Literal

File:line — `speculative.py:34-49` (the `MTPModelTypes = Literal[...]` block).
Insert `"gemma4_mtp",` immediately before the closing `]` (after the
`"step3p5_mtp",` entry on line 48).

```diff
@@ -34,6 +34,7 @@ MTPModelTypes = Literal[
     "longcat_flash_mtp",
     "mtp",
     "pangu_ultra_moe_mtp",
     "step3p5_mtp",
+    "gemma4_mtp",
 ]
```

Effect: `get_args(MTPModelTypes)` now contains `"gemma4_mtp"`, so the
auto-detect branch at :507-510 will resolve `method = "mtp"` once
`hf_config_override` has rewritten the model_type (HUNK 2).

---

## HUNK 2 — rewrite `gemma4_assistant` -> `gemma4_mtp` in `hf_config_override`

File:line — `speculative.py:341-344` (the `step3p5` block) through `:346` (the
`MistralLarge3` block). Insert the new block AFTER the `step3p5` block and
BEFORE the `MistralLarge3` block (i.e. between current lines 344 and 346).

Note: the plan said "after the qwen3_5 block / before the longcat block"; in the
actual 0.19.1 tree those are lines 324-333 (qwen3_5) and 334-339 (longcat). The
ordering among these independent `if hf_config.model_type == ...` guards is
immaterial (each is its own disjoint guard); placing it just before `return`
keeps it adjacent to the other model_type-rename blocks. Anchor chosen:
immediately after the `step3p5` block (:341-344), matching the 0.23.0 placement
relative to step3p5 (0.23.0 :512 sits right after its step3p5 :498-500).

```diff
@@ -341,6 +341,18 @@ def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
         if hf_config.model_type == "step3p5":
             hf_config.model_type = "step3p5_mtp"
             n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
             hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

+        if hf_config.model_type in ("gemma4_assistant", "gemma4_unified_assistant"):
+            hf_config.model_type = "gemma4_mtp"
+            text_config = getattr(hf_config, "text_config", hf_config)
+            # The assistant runs all decoder layers in a single forward
+            # call to produce one draft token, so n_predict=1.
+            # num_kv_shared_layers must be 0: cross-model KV sharing is
+            # set up by the proposer after model construction. The real
+            # field lives in text_config (top-level is absent), so zero it
+            # there; fall back to top-level only if text_config is missing.
+            if hasattr(text_config, "num_kv_shared_layers"):
+                text_config.num_kv_shared_layers = 0
+            else:
+                hf_config.update({"num_kv_shared_layers": 0})
+            hf_config.update({"n_predict": 1, "architectures": ["Gemma4MTPModel"]})
+
         if initial_architecture == "MistralLarge3ForCausalLM":
             hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

         return hf_config
```

Notes on adaptation from 0.23.0 (:512-521):
 - 0.23.0 uses `getattr(hf_config, "text_config", hf_config)` then
   `if hasattr(text_config, "num_kv_shared_layers"): text_config.num_kv_shared_layers = 0`.
   Kept verbatim. Added the explicit `else` top-level fallback (the plan's spec)
   for robustness — harmless because for THIS draft, `text_config` always exists
   and has the field (confirmed nksl=4), so the `if` branch fires.
 - `hf_config.update(...)` is the PretrainedConfig mutator used by every sibling
   block in this same function (0.19.1 :233, :247, :312, :328, :337, :344) — same
   API in 0.19.1, no signature change.
 - This runs as `hf_overrides=SpeculativeConfig.hf_config_override` during draft
   `ModelConfig` construction (:488), BEFORE the method auto-detect at :507-510.
   With model_type now `gemma4_mtp` (in `MTPModelTypes` via HUNK 1), :507-510
   sets `method = "mtp"`.
 - `n_predict=1` + `num_speculative_tokens=1` passes the divisibility check at
   :565-577 (`1 > 1` is False, so the raise path is skipped; and if
   `num_speculative_tokens` is left None it defaults to `n_predict=1` at :568).

---

## HUNK 3 — add `use_gemma4_mtp()` predicate

File:line — `speculative.py:859-860` (`use_eagle`). Insert the new method
immediately BEFORE `use_eagle` (so it is defined alongside the other
`use_*`/`uses_*` predicates) — its placement in the source is cosmetic, but the
RUNTIME ordering constraint is enforced by the runner dispatch (artifact #7),
not here.

CRITICAL ordering note (for the runner, artifact #7 / eagle.py): `use_eagle()`
at :860 returns True for `method == "mtp"`, which `gemma4_mtp` also satisfies.
So any dispatch must check `use_gemma4_mtp()` BEFORE `use_eagle()`.

Port of 0.23.0 :1054-1060 verbatim (the predicate has no 0.23.0-only API):

```diff
@@ -857,6 +857,14 @@ def num_lookahead_slots(self) -> int:
         return slots_per_req

+    def use_gemma4_mtp(self) -> bool:
+        return (
+            self.method == "mtp"
+            and self.draft_model_config is not None
+            and getattr(self.draft_model_config.hf_config, "model_type", None)
+            == "gemma4_mtp"
+        )
+
     def use_eagle(self) -> bool:
         return self.method in ("eagle", "eagle3", "mtp")
```

Adaptation check: `self.method`, `self.draft_model_config`, and
`draft_model_config.hf_config.model_type` all exist unchanged in 0.19.1
(`draft_model_config` is the attr set at :470-490; `use_eagle` at :859 already
reads `self.method`). No rename needed.

---

## Post-apply sanity

After applying all three hunks, the auto-resolution chain for the gemma4 draft is:
1. `hf_config_override` (:488 hook) rewrites model_type -> `gemma4_mtp`,
   text_config.num_kv_shared_layers -> 0, n_predict -> 1,
   architectures -> ["Gemma4MTPModel"].   [HUNK 2]
2. auto-detect (:507-510) sees `gemma4_mtp in get_args(MTPModelTypes)` -> `method="mtp"`.   [HUNK 1]
3. divisibility (:565-577) passes (n_predict=1).
4. `use_gemma4_mtp()` returns True; runner must branch on it before `use_eagle()`.   [HUNK 3]

No other lines in this file need to change.
