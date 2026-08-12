# Patch: registry.py — add Gemma4MTPModel speculative-decoding entry

**Target file:** `vllm/model_executor/models/registry.py`
**Tree:** 0.19.1 (`<vllm-src>/vllm/model_executor/models/registry.py`)
**Action:** patch (add one dict entry)

## What / why

Register the Gemma4 MTP draft so the spec-decode loader can resolve
`architectures=['Gemma4MTPModel']` (set by artifact #4, the model-arch-config
convertor HUNK 2) to module `gemma4_mtp`, class `Gemma4MTP` (artifact #1,
`vllm/model_executor/models/gemma4_mtp.py`).

Matches upstream 0.23.0 exactly:
`vllm-0.23.0-vanilla/vllm/model_executor/models/registry.py:627`
→ `"Gemma4MTPModel": ("gemma4_mtp", "Gemma4MTP"),`

## Insertion point (0.19.1)

`_SPECULATIVE_DECODING_MODELS` opens at line 545 and closes at line 579.
The Qwen3_5MTP entries are at lines 574-575; the disabled-models comment block
begins at line 576. Insert the new entry between line 575 and line 576 (i.e.
after the last live `Qwen3_5MoeMTP` entry, before the `# Temporarily disabled.`
comment). Dict ordering is not load-bearing — placement after the Qwen3_5MTP
entries follows the architect's spec.

## Unified diff

```diff
--- a/vllm/model_executor/models/registry.py
+++ b/vllm/model_executor/models/registry.py
@@ -572,7 +572,8 @@ _SPECULATIVE_DECODING_MODELS = {
     "Qwen3NextMTP": ("qwen3_next_mtp", "Qwen3NextMTP"),
     "Step3p5MTP": ("step3p5_mtp", "Step3p5MTP"),
     "Qwen3_5MTP": ("qwen3_5_mtp", "Qwen3_5MTP"),
     "Qwen3_5MoeMTP": ("qwen3_5_mtp", "Qwen3_5MoeMTP"),
+    "Gemma4MTPModel": ("gemma4_mtp", "Gemma4MTP"),
     # Temporarily disabled.
     # # TODO(woosuk): Re-enable this once the MLP Speculator is supported in V1.
     # "MLPSpeculatorPreTrainedModel": ("mlp_speculator", "MLPSpeculator"),
```

## Exact line to add

After `vllm/model_executor/models/registry.py:575`
(`    "Qwen3_5MoeMTP": ("qwen3_5_mtp", "Qwen3_5MoeMTP"),`):

```python
    "Gemma4MTPModel": ("gemma4_mtp", "Gemma4MTP"),
```

(4-space indent, trailing comma, consistent with the surrounding dict entries.)

## Cross-artifact consistency

- Module key `"gemma4_mtp"` resolves relative to `vllm.model_executor.models`
  (same convention as `qwen3_5_mtp`, `deepseek_mtp`, etc.) → maps to
  `vllm/model_executor/models/gemma4_mtp.py` (artifact #1).
- Class `"Gemma4MTP"` must match the class name defined in artifact #1.
- Architecture key `"Gemma4MTPModel"` must match the string written into
  `config.architectures` by artifact #4 (model_arch_config_convertor.py HUNK 2).
- NOTE: 0.23.0 uses bare module name `"gemma4_mtp"` (not the fully-qualified
  `"vllm.models.deepseek_v4"` form seen at 0.23.0:626 for DeepSeekV4MTP). We
  follow the bare form, identical to 0.23.0:627 and to every existing 0.19.1
  `_SPECULATIVE_DECODING_MODELS` entry — the registry loader prepends the
  `vllm.model_executor.models` package path for non-dotted module keys.
```
