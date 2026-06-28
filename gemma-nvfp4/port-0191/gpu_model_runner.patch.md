# Patch: vllm/v1/worker/gpu_model_runner.py (0.19.1)

Target file: `<vllm-src>/vllm/v1/worker/gpu_model_runner.py`

Adds `Gemma4Proposer` dispatch, the per-group block-table call site, and widens 5 `isinstance` asserts (+ 1 selector) that the Gemma4 MTP path enters because `SpeculativeConfig.use_gemma4_mtp()` makes `use_eagle()` return True for `method=='mtp'`.

All line numbers reference the unpatched 0.19.1 file as read. Apply hunks top-to-bottom; later line numbers shift after earlier insertions, so the context lines (not the numbers) are authoritative.

---

## HUNK 1 — import (after line 164)

```diff
 from vllm.v1.spec_decode.draft_model import DraftModelProposer
 from vllm.v1.spec_decode.eagle import EagleProposer
+from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
 from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
```

Context: line 163 `from vllm.v1.spec_decode.draft_model import DraftModelProposer`, line 164 `from vllm.v1.spec_decode.eagle import EagleProposer`, line 165 `from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer`. Insert the new import on its own line between 164 and 165.

---

## HUNK 2 — drafter dispatch (insert BEFORE line 551 `elif self.speculative_config.use_eagle():`)

Rationale: `use_gemma4_mtp()` (added in speculative.py patch) returns True only for the gemma4 MTP method; but `use_eagle()` ALSO returns True for `method=='mtp'`, so the gemma4 branch MUST come first or `use_eagle()` swallows it.

Existing 0.19.1 dispatch tail (lines 549-565):

```python
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
            elif self.speculative_config.method == "medusa":
```

Patched:

```diff
             elif self.speculative_config.method == "suffix":
                 self.drafter = SuffixDecodingProposer(self.vllm_config)
+            elif self.speculative_config.use_gemma4_mtp():
+                self.drafter = Gemma4Proposer(self.vllm_config, self.device, self)
             elif self.speculative_config.use_eagle():
                 self.drafter = EagleProposer(self.vllm_config, self.device, self)
                 if self.speculative_config.method == "eagle3":
                     self.use_aux_hidden_state_outputs = (
                         self.drafter.eagle3_use_aux_hidden_state
                     )
             elif self.speculative_config.method == "medusa":
```

Note the type-annotation union at lines 514-521 declares `self.drafter` (NgramProposer | SuffixDecodingProposer | EagleProposer | DraftModelProposer | MedusaProposer | ExtractHiddenStatesProposer). This is a type hint only (not runtime-enforced); the new `Gemma4Proposer` assignment is valid Python regardless. Optionally widen it for type-checker cleanliness:

```diff
                 | DraftModelProposer
                 | MedusaProposer
+                | Gemma4Proposer
                 | ExtractHiddenStatesProposer
             )
```

(Optional — runtime-safe either way; matching the spec's runtime-correctness scope, the asserts in HUNK 4 are the load-bearing ones.)

---

## HUNK 3 — per-kv-group loop: widen selector + add per-group block-table call (lines 2306-2313)

Mirrors 0.23.0 :2425-2447 (`set_per_group_block_table` call site). 0.19.1 uses `self.drafter.kv_cache_gid` (singular attr); gemma4.py sets `self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id` (gemma4.py:267), so the existing `kv_cache_gid == kv_cache_gid` guard at the selected line works once `Gemma4Proposer` is added to the isinstance tuple.

Existing 0.19.1 (lines 2306-2313):

```python
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, EagleProposer):
                    if self.drafter.kv_cache_gid == kv_cache_gid:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
```

Patched:

```diff
             if self.speculative_config and spec_decode_common_attn_metadata is None:
-                if isinstance(self.drafter, EagleProposer):
+                if isinstance(self.drafter, (EagleProposer, Gemma4Proposer)):
                     if self.drafter.kv_cache_gid == kv_cache_gid:
                         spec_decode_common_attn_metadata = cm
                 else:
                     spec_decode_common_attn_metadata = cm
+
+            # Capture per-group block tables for the Gemma4 multi-group proposer.
+            if self.speculative_config and isinstance(self.drafter, Gemma4Proposer):
+                self.drafter.set_per_group_block_table(
+                    kv_cache_gid, cm.block_table_tensor
+                )
 
             for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
```

The new `set_per_group_block_table` block is inserted AFTER the selector `if/else` (after the original line 2311) and BEFORE `for attn_gid in range(...)` (original line 2313), inside the `for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):` loop so it fires once per group. `cm.block_table_tensor` was already reassigned for `kv_cache_gid > 0` at lines 2302-2304, so the per-group tensor is correct.

---

## HUNK 4 — isinstance-union asserts (5 sites)

Each guard below evaluates True for `method=='mtp'` because `use_eagle()` returns True for gemma4 MTP. The asserts use the PEP-604 `X | Y` runtime-union form (valid as second arg to `isinstance`). Add `| Gemma4Proposer` to each.

### 4a — line 4218-4221 (use_gpu_toks path; guard at 4210-4214 includes `use_eagle()`)

```diff
                 assert isinstance(
                     self.drafter,
-                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
+                    EagleProposer
+                    | DraftModelProposer
+                    | ExtractHiddenStatesProposer
+                    | Gemma4Proposer,
                 )
```

### 4b — line 4606 (guard at 4605 `elif spec_config.use_eagle() or spec_config.uses_draft_model():`)

```diff
         elif spec_config.use_eagle() or spec_config.uses_draft_model():
-            assert isinstance(self.drafter, EagleProposer | DraftModelProposer)
+            assert isinstance(
+                self.drafter, EagleProposer | DraftModelProposer | Gemma4Proposer
+            )
```

### 4c — line 5492-5495 (cudagraph branch; guard at 5487-5491 includes `use_eagle()`)

```diff
                 assert isinstance(
                     self.drafter,
-                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
+                    EagleProposer
+                    | DraftModelProposer
+                    | ExtractHiddenStatesProposer
+                    | Gemma4Proposer,
                 )
```

### 4d — line 6247 (CRITICAL — guards `initialize_attn_backend` at 6248; guard at 6243-6246 includes `use_eagle()`)

```diff
         if self.speculative_config and (
             self.speculative_config.use_eagle()
             or self.speculative_config.uses_draft_model()
         ):
-            assert isinstance(self.drafter, EagleProposer | DraftModelProposer)
+            assert isinstance(
+                self.drafter, EagleProposer | DraftModelProposer | Gemma4Proposer
+            )
             self.drafter.initialize_attn_backend(kv_cache_config, kernel_block_sizes)
```

This is the site where the Gemma4 multi-group attn-backend override fires (the proposer's `initialize_attn_backend` builds `draft_attn_groups` and sets `kv_cache_gid`). A missing addition here is a hard AssertionError before bring-up reaches any decode.

### 4e — line 6422 (guards `initialize_cudagraph_keys` at 6423; guard at 6418-6420 includes `use_eagle()`)

```diff
         if self.speculative_config and (
             self.speculative_config.use_eagle()
             or self.speculative_config.uses_extract_hidden_states()
         ):
-            assert isinstance(self.drafter, EagleProposer | ExtractHiddenStatesProposer)
+            assert isinstance(
+                self.drafter,
+                EagleProposer | ExtractHiddenStatesProposer | Gemma4Proposer,
+            )
             self.drafter.initialize_cudagraph_keys(cudagraph_mode)
```

---

## Dependency notes (cross-file, for apply order)

- HUNK 1/2/3/4 require `vllm/v1/spec_decode/gemma4.py` to exist and export `Gemma4Proposer` with: attr `kv_cache_gid` (gemma4.py:267), methods `set_per_group_block_table(kv_cache_gid, block_table_tensor)`, `initialize_attn_backend(kv_cache_config, kernel_block_sizes)`, `initialize_cudagraph_keys(cudagraph_mode)`. Verify these signatures match the created gemma4.py artifact before shipping.
- HUNK 2 requires `SpeculativeConfig.use_gemma4_mtp()` to exist (speculative.py patch) AND `use_eagle()` to return True for `method=='mtp'` — confirm the speculative.py patch keeps `use_eagle()` truthy for gemma4 so all five guards in HUNK 4 are entered (the patch design relies on that).
- 0.19.1 has NO `DFlashProposer` / `Step3p5MTPProposer` (present in 0.23.0 :2430/:2440); do NOT import them. The 0.23.0 selector tuple included them; the 0.19.1 selector only needs `(EagleProposer, Gemma4Proposer)`.

## Verification after apply

```bash
python3 -c "import ast; ast.parse(open('<vllm-src>/vllm/v1/worker/gpu_model_runner.py').read())"
grep -n "Gemma4Proposer" <vllm-src>/vllm/v1/worker/gpu_model_runner.py
# expect: 1 import + 1 dispatch + 1 selector + 1 set_per_group_block_table call + 5 assert-union sites = 9 (10 if optional type-hint union added)
```
