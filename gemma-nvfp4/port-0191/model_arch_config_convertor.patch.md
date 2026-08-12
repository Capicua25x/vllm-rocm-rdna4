# Patch: model_arch_config_convertor.py — gemma4_mtp backbone-hidden-size convertor

Target file (0.19.1):
`<vllm-src>/vllm/transformers_utils/model_arch_config_convertor.py`

## Why

The speculator/EAGLE runner sizes the **hidden_states feedback buffer** from the
draft model's config:

- `eagle.py:83` → `self.hidden_size = self.draft_model_config.get_hidden_size()`

The base `get_hidden_size` returns the draft's *text* hidden size:

- base `get_hidden_size` (`model_arch_config_convertor.py:45-46`):
  `return getattr(self.hf_text_config, "hidden_size", 0)` → for this draft = **1024**.

But the Gemma4 MTP draft's attention is **Q-only** and shares the target's KV; its
`post_projection.weight[2816,1024]` projects draft hidden (1024) back up to the
**backbone** dim, and `_maybe_share_embeddings` swaps in the 2816 target embedding.
So the hidden_states feedback buffer must be **backbone_hidden_size = 2816**, NOT 1024.

If left at 1024 → shape mismatch / corrupt drafts with **no clean error**
(silently-wrong if omitted). The draft config carries `backbone_hidden_size=2816`
at top level (`self.hf_config`), so the override reads `self.hf_config.backbone_hidden_size`.

This ports 0.23.0 vanilla `Gemma4MTPModelArchConfigConvertor`
(`vllm-0.23.0-vanilla/.../model_arch_config_convertor.py:552-561`) into 0.19.1.

Note on adaptation: 0.23.0's neighboring `Gemma4ModelArchConfigConvertor` differs from
0.19.1's (0.23.0 uses `is_mm_prefix_lm`; 0.19.1 uses `get_head_size`). We do NOT touch
the existing 0.19.1 `Gemma4ModelArchConfigConvertor`; we only add the MTP subclass and
register it. The ported MTP class itself is API-identical between versions (base
`get_hidden_size`/`get_num_hidden_layers` signatures match 0.19.1 at :45-46 / :36-37).

---

## HUNK 1 — add `Gemma4MTPModelArchConfigConvertor`

Insert the new class immediately **before** the existing
`Gemma4ModelArchConfigConvertor` (which begins at 0.19.1 line 451). Context shown:
the preceding `LongCatFlashMTPModelArchConfigConvertor` (ends :448) and the start of
`Gemma4ModelArchConfigConvertor` (:451).

```diff
@@ vllm/transformers_utils/model_arch_config_convertor.py @@ (around lines 446-451)
 class LongCatFlashMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
     def get_num_hidden_layers(self) -> int:
         return getattr(self.hf_text_config, "num_nextn_predict_layers", 1)
 
 
+class Gemma4MTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
+    def get_hidden_size(self) -> int:
+        # The speculator buffer must match the backbone (target) model's
+        # hidden dimension, not the draft model's smaller dimension.
+        return getattr(
+            self.hf_config, "backbone_hidden_size", super().get_hidden_size()
+        )
+
+    def get_num_hidden_layers(self) -> int:
+        return getattr(self.hf_text_config, "num_hidden_layers", 0)
+
+
 class Gemma4ModelArchConfigConvertor(ModelArchConfigConvertorBase):
     def get_head_size(self) -> int:
```

---

## HUNK 2 — register `gemma4_mtp` in `MODEL_ARCH_CONFIG_CONVERTORS`

Add the dict entry **after** the `gemma4` / `gemma4_text` entries (0.19.1 lines
484-485). The dict currently closes at line 486 (`}`).

```diff
@@ vllm/transformers_utils/model_arch_config_convertor.py @@ (around lines 484-486)
     "gemma4": Gemma4ModelArchConfigConvertor,
     "gemma4_text": Gemma4ModelArchConfigConvertor,
+    "gemma4_mtp": Gemma4MTPModelArchConfigConvertor,
 }
```

(`gemma4_mtp` is the draft checkpoint's `config.model_type`.)

---

## Verification after apply

- `grep -n "gemma4_mtp" model_arch_config_convertor.py` → one class def, one dict entry.
- `python3 -c "import ast; ast.parse(open('.../model_arch_config_convertor.py').read())"` → OK.
- Runtime sanity: with the draft loaded, `eagle.py:83` `self.hidden_size` must equal
  **2816** (= `backbone_hidden_size`), not 1024.
