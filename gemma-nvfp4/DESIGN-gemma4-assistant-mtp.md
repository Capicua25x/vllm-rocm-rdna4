> ⚠️ **SUPERSEDED 2026-06-25 — do not implement the v1/v2 reverse-engineering plan below.**
> vLLM **0.23.0 already ships the complete, correct implementation** (`model_executor/models/gemma4_mtp.py`
> + `v1/spec_decode/gemma4.py`), including the shared-KV "v2" path this doc deferred as upstream-grade.
> The current plan is to **backport 0.23.0 → the 0.19.1 RDNA4 image**: see **`port-0191/PORT-PLAN.md`**
> (verdict: GO, moderate-adapt) and the staged, AST-validated artifacts in `port-0191/`.
> The reverse-engineered scaffold `gemma4_assistant.vllm.py` is **falsified** — the real draft is **Q-only
> KV-shared** (no k/v weights; `pre_projection` is `2·backbone=5632→1024`), so it cannot run causally.
> This doc is kept for history only.

# Native vLLM `Gemma4AssistantForCausalLM` — MTP-style spec-decode design (target: MTP3-class)

Goal: serve `google/gemma-4-26B-A4B-it-assistant` as a vLLM speculative-decode draft for the
`RedHatAI/gemma-4-26B-A4B-it-NVFP4` target on RDNA4, reaching ~MTP3-class acceptance — to close the
single-stream gap (Gemma no-spec ≈ 50–66 tok/s vs Qwen-MTP ~92). Status: **DESIGNED + v1 scaffold**;
not yet load-tested (see "Reality" at bottom).

## Why this is MTP/eagle, NOT a `--speculative-config model=` draft
The bring-up attempt that used `--speculative-config {"model": "...assistant"}` routed through
`DraftModelProposer` (`pass_hidden_states_to_model=False`) — a *standalone* draft that never receives
the target's hidden states. But `gemma4_assistant.forward` **requires** the target's hidden states
(`inputs_embeds` is `2×backbone_hidden` = target-hidden ⊕ token-embed via `pre_projection`) AND the
target's last-layer shared KV (`shared_kv_states` per layer-type). So it must be an **MTP/eagle** model:
vLLM's `SpecDecodeBaseProposer.propose()` feeds `target_hidden_states` → `combine_hidden_states()` →
the draft. Register it in `_SPECULATIVE_DECODING_MODELS` and select the MTP method.

## Template: `qwen3_5_mtp.py::Qwen3_5MTP` (what the Qwen MTP path uses — near 1:1 map)
| Qwen3_5MTP piece | Gemma4Assistant equivalent |
|---|---|
| `Qwen3_5MultiTokenPredictor.fc` (`ColumnParallelLinear(hidden*2, hidden)`) | `pre_projection` (`Linear(2*backbone_hidden, hidden)`) |
| `embed_tokens` (`VocabParallelEmbedding`) | same (draft's text_config vocab/hidden) |
| `layers` (MTP decoder layers) | the draft's small **Gemma4 backbone** — reuse `gemma4.py::Gemma4DecoderLayer` (draft `text_config.num_hidden_layers`, small for 419M) |
| `pre_fc_norm_{hidden,embedding}`, `norm` (RMSNorm) | Gemma4 RMSNorm (reuse `gemma4.py`); gemma also applies `embed_scale` (×√hidden) |
| forward: `cat([norm(embed), norm(target_hidden)])` → `fc` → layer → `norm` | same shape; **the combine** |
| `Qwen3_5MTP.compute_logits` (lm_head) | **`post_projection`(hidden→backbone)** then lm_head (or centroid head) |

## v1 (load-testable) — defer shared-KV, condition on target hidden states only
Implement `Gemma4AssistantForCausalLM(nn.Module)` + inner predictor, exactly like `Qwen3_5MTP`:
- `combine_hidden_states(target_hidden)` → `pre_projection` input half. **Open Q (resolve at bring-up):**
  the Transformers ref `pre_projection` is `2*backbone_hidden` wide and the *caller* builds the 2×
  vector — confirm the two halves (target-hidden ⊕ prev-token-embed? ⊕ target-hidden-shifted?). v1:
  follow the MTP template (`cat([token_embed, target_hidden])`), fix against the ref if acceptance is ~0.
- backbone: a small `Gemma4Model` over the draft `text_config` (reuse `Gemma4DecoderLayer`), run
  **causally, no shared-KV cross-attention** (this is the v1 simplification — lower acceptance, but it
  loads + generates on the existing MTP proposer).
- `compute_logits`: `post_projection(last_hidden)` → `lm_head`. Skip the centroid `masked_embedding`
  for v1 (`use_ordered_embeddings` path) — add in v2.
- `load_weights`: map draft checkpoint names → vLLM params. Names from the draft snapshot:
  `model.*` (Gemma4 backbone), `lm_head.weight` (tied to `model.embed_tokens.weight`),
  `pre_projection.weight`, `post_projection.weight`, `masked_embedding.{centroids.weight,token_ordering}`.
  Reuse `AutoWeightsLoader` + the gemma4 stacked/expert mappings (it's an A4B MoE backbone → needs the
  RDNA NVFP4 MoE path too, OR the draft is dense — CHECK the draft's text_config for `num_experts`).
- Register `"Gemma4AssistantForCausalLM": ("gemma4_assistant", "Gemma4AssistantForCausalLM")` in
  `registry.py::_SPECULATIVE_DECODING_MODELS`.
- Serve: `--speculative-config '{"model":"google/gemma-4-26B-A4B-it-assistant","method":"mtp","num_speculative_tokens":3}'`
  (method must resolve to the MTP proposer; if auto-detect fails on `model_type=gemma4_assistant`, set
  `method` explicitly or add a config shim). **No `--language-model-only` needed** on this path (MTP
  proposer doesn't run the target-multimodal vision-encoder probe that killed the draft-model path).

## v2 (reach MTP3-class acceptance) — the shared-KV cross-attention
The acceptance comes from the draft cross-attending to the target's last-layer KV (`shared_kv_states`
for `full_attention` + `sliding_attention`) with the bidirectional/SWA-flip masks
(`create_attention_masks`). vLLM's spec framework feeds hidden states but **not** per-layer-type shared
KV to the draft → needs runner plumbing:
1. Capture the target's last-layer KV per layer-type during the target forward.
2. Pass them to the proposer → into the draft's attention as cross-attn KV.
3. Implement the bidirectional + SWA-flip mask construction in vLLM attention.
This is framework-level (an upstream-PR-grade change). v1 proves the model + pipeline; v2 is where the
acceptance climbs toward MTP3.

## Bring-up loop (when resumed — GPU window)
1. bind-mount `gemma4_assistant.py` → `/build/vllm/vllm/model_executor/models/gemma4_assistant.py`
   + a patched `registry.py` (add the `_SPECULATIVE_DECODING_MODELS` entry).
2. Launch target (the confirmed `launch-26b-019-triton.sh` config) **+** the spec-config above.
3. Iterate: weight-load KeyErrors → fix mapping; forward shape mismatch → fix combine width;
   `method` not resolving → set explicitly. Watch `docker logs` for the spec acceptance-rate line.
4. If acceptance ~0 with v1, the combine-semantics or the missing shared-KV is the cause → v2.

## Reality / honest scope
v1 is a real, load-testable artifact and a few GPU iterations from "loads + generates with *some*
acceptance." **MTP3-class acceptance requires v2 (shared-KV framework plumbing)** — genuinely
multi-session, upstream-grade. Impact reminder: this only improves *single-stream* latency; at
concurrency Gemma already matches Qwen — so this
is an optimization, not a blocker. Recommend executing v1→v2 as a dedicated session with this doc as
the spec; the coding is offline (GPU only for the load-tests).
