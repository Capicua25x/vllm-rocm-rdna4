# Patch: `vllm/v1/spec_decode/eagle.py` (0.19.1)

Base `SpecDecodeBaseProposer` refactor for the Gemma4-MTP backport.

- **Target file (0.19.1):** `<vllm-src>/vllm/v1/spec_decode/eagle.py`
- **Hook body source (0.23.0):** `vllm-0.23.0-vanilla/vllm/v1/spec_decode/llm_base_proposer.py:881-893`
- **Caller-pattern source (0.23.0):** `vllm-0.23.0-vanilla/vllm/v1/spec_decode/llm_base_proposer.py:483-485`

Scope/limits per architect:
- Behavior-preserving for `EagleProposer` / Qwen-MTP: the new hook produces the same `per_layer_attn_metadata` dict (identical loop), and now also returns the per-group list (currently discarded by `propose()` via `_,`).
- `propose_tree` (`eagle.py:1028`) is NOT touched — dead at `n_predict=1`.
- The multi-step `constant_draft_positions` branches are NOT backported (only needed when `num_speculative_tokens > 1`); only the forward-compat default attribute is added so the new `gemma4.py` / `gemma4_mtp.py` can reference `self.constant_draft_positions` without `AttributeError`.
- `CommonAttentionMetadata` is already imported at `eagle.py:30`, so the hook's type annotation needs no new import.

---

## HUNK 1 — replace inlined per-group build loop in `propose()` (eagle.py:426-432)

Context: this sits between the `set_inputs_first_pass(...)` block (ends line 424) and the
`_determine_batch_execution_and_padding(...)` call (begins line 434).

### Before (eagle.py:424-436)

```python
        )

        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=0
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
```

### After

```python
        )

        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
            common_attn_metadata
        )

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
```

### Unified-diff form

```diff
@@ eagle.py:426-432 (in SpecDecodeBaseProposer.propose, after set_inputs_first_pass) @@
         )

-        per_layer_attn_metadata: dict[str, object] = {}
-        for attn_group in self.draft_attn_groups:
-            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
-                common_attn_metadata=common_attn_metadata, draft_index=0
-            )
-            for layer_name in attn_group.layer_names:
-                per_layer_attn_metadata[layer_name] = attn_metadata
+        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
+            common_attn_metadata
+        )

         cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
             self._determine_batch_execution_and_padding(num_tokens)
         )
```

---

## HUNK 2 — add the `build_per_group_and_layer_attn_metadata` method to `SpecDecodeBaseProposer`

Body is verbatim from 0.23.0 `llm_base_proposer.py:881-893` (default `draft_index=0` preserves the
old `draft_index=0` call). Place it inside class `SpecDecodeBaseProposer`. A safe insertion point is
immediately after the `propose()`/inlined-build code, before the next def — e.g. right after the end of
`propose()`. Indentation = 4 spaces (method level). Any location inside the class body works since it
only references `self.draft_attn_groups`; recommended: insert directly above the existing
`_determine_batch_execution_and_padding` definition (find it with `grep -n "def _determine_batch_execution_and_padding" eagle.py`).

### New method (add)

```python
    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=draft_index
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata
```

### Unified-diff form (example anchor: insert before `def _determine_batch_execution_and_padding`)

```diff
@@ SpecDecodeBaseProposer class body @@
+    def build_per_group_and_layer_attn_metadata(
+        self,
+        common_attn_metadata: CommonAttentionMetadata,
+        draft_index: int = 0,
+    ) -> tuple[list[object], dict[str, object]]:
+        per_group_attn_metadata: list[object] = []
+        per_layer_attn_metadata: dict[str, object] = {}
+        for attn_group in self.draft_attn_groups:
+            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
+                common_attn_metadata=common_attn_metadata, draft_index=draft_index
+            )
+            per_group_attn_metadata.append(attn_metadata)
+            for layer_name in attn_group.layer_names:
+                per_layer_attn_metadata[layer_name] = attn_metadata
+        return per_group_attn_metadata, per_layer_attn_metadata
+
     def _determine_batch_execution_and_padding(self, num_tokens):
```

Note: the type annotation uses `CommonAttentionMetadata`, already imported at `eagle.py:30`
(`from vllm.v1.attention.backend import CommonAttentionMetadata`). No new import needed. The
0.23.0 original returns `tuple[list[object], dict[str, object]]` — kept identical.

---

## HUNK 3 — add forward-compat `constant_draft_positions` default in `__init__`

Insert immediately after `self.pass_hidden_states_to_model = pass_hidden_states_to_model`
(eagle.py:72). Indentation = 8 spaces (inside `__init__`).

### Before (eagle.py:72)

```python
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
```

### After

```python
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        # Forward-compat default (0.23.0 parity). Multi-step constant-position
        # drafting branches are NOT backported; this attribute only exists so
        # gemma4 spec-decode plumbing can read it without AttributeError.
        # Stays False for the n_predict=1 Gemma4-MTP target path.
        self.constant_draft_positions: bool = False
```

### Unified-diff form

```diff
@@ eagle.py:72 (SpecDecodeBaseProposer.__init__) @@
         self.pass_hidden_states_to_model = pass_hidden_states_to_model
+        # Forward-compat default (0.23.0 parity). Multi-step constant-position
+        # drafting branches are NOT backported; this attribute only exists so
+        # gemma4 spec-decode plumbing can read it without AttributeError.
+        # Stays False for the n_predict=1 Gemma4-MTP target path.
+        self.constant_draft_positions: bool = False
```

---

## Apply checklist

1. HUNK 1: line range 426-432 replaced (net -7 lines, +3 lines). Verify the surrounding
   `set_inputs_first_pass(...)` close-paren (424) and `_determine_batch_execution_and_padding`
   call (434) still bracket the replacement.
2. HUNK 2: new method added to class `SpecDecodeBaseProposer`. Confirm it is inside the class
   (4-space indent) and NOT after the class ends. `grep -n "def build_per_group_and_layer_attn_metadata" eagle.py`
   should return exactly one hit at method-level indentation.
3. HUNK 3: `self.constant_draft_positions` set in `__init__` after line 72. `grep -n "constant_draft_positions" eagle.py`
   should show one assignment (no other references in this file — multi-step branches intentionally omitted).
4. `python3 -c "import ast; ast.parse(open('eagle.py').read())"` after applying.
