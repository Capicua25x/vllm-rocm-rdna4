# SPDX-License-Identifier: Apache-2.0
#
# ⚠️ SUPERSEDED / DO-NOT-USE (2026-06-25). This reverse-engineered scaffold is FALSIFIED:
#   the real google/gemma-4-26B-A4B-it-assistant draft is Q-ONLY KV-SHARED (self_attn has only
#   q_proj/q_norm/o_proj — NO k_proj/v_proj), so it CANNOT run causally over reused Gemma4DecoderLayer,
#   and pre_projection is Linear(2*backbone=5632 -> 1024) requiring the TARGET's 2816-dim embedding
#   (embedding-sharing), not cat([token_embed(1024), target_hidden]). vLLM 0.23.0 already implements the
#   correct model (gemma4_mtp.py) + proposer (v1/spec_decode/gemma4.py). The live plan is the
#   0.23.0->0.19.1 backport: see ../port-0191/PORT-PLAN.md + the staged artifacts in ../port-0191/.
#   Kept for history only.
#
# Native vLLM Gemma-4 Assistant (MTP-style spec-decode draft) — v1 SCAFFOLD, NOT YET LOAD-TESTED.
#
# Drop into vllm/model_executor/models/gemma4_assistant.py and register in registry.py:
#   _SPECULATIVE_DECODING_MODELS["Gemma4AssistantForCausalLM"] = ("gemma4_assistant", "Gemma4AssistantForCausalLM")
#
# Mapped 1:1 onto qwen3_5_mtp.py::Qwen3_5MTP (the pattern the Qwen MTP path uses). See
# DESIGN-gemma4-assistant-mtp.md for the full design, the v2 shared-KV plan, and the bring-up loop.
#
# v1 SIMPLIFICATIONS (deliberate, to get a load-testable artifact):
#   * NO shared-KV cross-attention to the target (the Transformers ref cross-attends to the target's
#     last-layer {full,sliding} KV). v1 runs the small Gemma4 backbone causally on the projected
#     combine vector. This is the v2 work that lifts acceptance toward MTP3 (needs runner plumbing).
#   * NO centroid `masked_embedding` logit head (use_ordered_embeddings) — v1 uses post_projection→lm_head.
#   * Draft backbone is assumed DENSE bf16 (419M → no 256-expert MoE; CONFIRM via draft text_config
#     num_experts at bring-up). If MoE, swap Gemma4DecoderLayer's MoE path + RDNA NVFP4 (unlikely at 419M).
#
# ── BRING-UP TODOs (resolve with load-tests; each is a one-iteration fix) ─────────────────────────
#   TODO-1 COMBINE: confirm the two halves of pre_projection's 2*backbone_hidden input. v1 follows the
#          MTP template: cat([token_embed, target_hidden]). If acceptance ~0, check the Transformers
#          ref caller (generate w/ assistant_model) for the true construction.
#   TODO-2 WIDTHS: target_hidden is backbone_hidden wide; the draft's hidden_size may differ. The
#          pre_projection maps 2*backbone_hidden→hidden; post_projection maps hidden→backbone_hidden
#          (the proposer feeds last_hidden_state back as the next-step target hidden — so it must be
#          backbone_hidden wide; that's why post_projection exists). Wire combine/return accordingly.
#   TODO-3 WEIGHTS: draft checkpoint names → vLLM params. Expected: model.* (Gemma4 backbone),
#          lm_head.weight (tied to model.embed_tokens.weight), pre_projection.weight, post_projection.weight,
#          masked_embedding.{centroids.weight,token_ordering}. Verify with the snapshot; extend remap below.
#   TODO-4 METHOD: ensure vLLM selects the MTP proposer for this model (config method='mtp' or a
#          model_type→method shim) so propose() feeds target_hidden_states to combine_hidden_states.

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer  # reuse the dense backbone layer
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors


class Gemma4AssistantPredictor(nn.Module):
    """The draft: a small Gemma4 backbone conditioned on the target hidden state (v1: no shared-KV)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        text_cfg = cfg.get_text_config()
        self.config = cfg
        self.text_config = text_cfg
        self.hidden_size = text_cfg.hidden_size
        self.backbone_hidden_size = cfg.backbone_hidden_size
        self.vocab_size = text_cfg.vocab_size

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        # gemma scales embeddings by sqrt(hidden) — match Gemma4Model (embed_scale).
        self.embed_scale = self.hidden_size**0.5

        # pre_projection == Qwen3_5MTP.fc, but 2*backbone_hidden (not 2*hidden) → hidden.  TODO-1/2
        self.pre_projection = ColumnParallelLinear(
            2 * self.backbone_hidden_size, self.hidden_size,
            bias=False, gather_output=True, return_bias=False,
            prefix=f"{prefix}.pre_projection",
        )
        # post_projection: draft hidden → backbone_hidden (fed back to the proposer as next target hidden). TODO-2
        self.post_projection = ColumnParallelLinear(
            self.hidden_size, self.backbone_hidden_size,
            bias=False, gather_output=True, return_bias=False,
            prefix=f"{prefix}.post_projection",
        )

        # Small dense Gemma4 backbone (draft text_config.num_hidden_layers — small for 419M).
        n = text_cfg.num_hidden_layers
        self.layers = nn.ModuleList(
            Gemma4DecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}") for i in range(n)
        )
        self.norm = RMSNorm(self.hidden_size, eps=text_cfg.rms_norm_eps)
        # Normalize each half before concat (mirrors qwen3_5 pre_fc_norm_{hidden,embedding}).
        self.pre_norm_embed = RMSNorm(self.backbone_hidden_size, eps=text_cfg.rms_norm_eps)
        self.pre_norm_hidden = RMSNorm(self.backbone_hidden_size, eps=text_cfg.rms_norm_eps)

    def combine(self, token_embed_b: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        # TODO-1/2: both halves are backbone_hidden wide; token_embed must be projected/sized to
        # backbone_hidden upstream (or here). v1 mirrors the MTP template's cat+norm+fc.
        x = torch.cat(
            [self.pre_norm_embed(token_embed_b), self.pre_norm_hidden(target_hidden)], dim=-1
        )
        return self.pre_projection(x)

    def forward(self, input_ids, positions, target_hidden, *, spec_step_idx: int = 0):
        token_embed = self.embed_tokens(input_ids) * self.embed_scale
        # NOTE: token_embed is hidden-wide; target_hidden is backbone-wide. The combine expects both
        # backbone-wide — TODO-2: reconcile (project token_embed→backbone, or the ref concatenates raw).
        hidden = self.combine(token_embed, target_hidden)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions=positions, hidden_states=hidden, residual=residual)
        hidden, _ = self.norm(hidden, residual)
        return hidden  # draft-hidden; the wrapper post_projects for the proposer + lm_head for logits


class Gemma4AssistantForCausalLM(nn.Module):
    """MTP-style wrapper — interface parallels qwen3_5_mtp.Qwen3_5MTP (combine_hidden_states/forward/compute_logits)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        text_cfg = cfg.get_text_config()
        self.config = cfg
        self.model = Gemma4AssistantPredictor(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if text_cfg.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                text_cfg.vocab_size, text_cfg.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
            )
        self.logits_processor = LogitsProcessor(text_cfg.vocab_size)

    # The proposer calls this on the target hidden states before forward (see eagle.py propose()). TODO-4
    def combine_hidden_states(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return target_hidden  # identity here; the real combine (with token embed) happens in forward()

    def forward(self, input_ids, positions, hidden_states, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        draft_hidden = self.model(input_ids, positions, hidden_states)
        # post_projection → backbone_hidden, returned so the proposer can feed it as the next-step
        # target hidden (multi-token). compute_logits uses the pre-projection draft_hidden for lm_head.
        self._last_draft_hidden = draft_hidden
        return self.model.post_projection(draft_hidden)

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        # v1: lm_head over the draft hidden (TODO: centroid masked_embedding when use_ordered_embeddings).
        return self.logits_processor(self.lm_head, getattr(self, "_last_draft_hidden", hidden_states))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap(ws):
            for name, w in ws:
                # TODO-3: confirm draft checkpoint prefixes. Expected top-level: model.*, lm_head.*,
                # pre_projection.*, post_projection.*, masked_embedding.*  → map model.* under self.model.
                if name.startswith("model."):
                    yield name, w
                elif name.startswith(("pre_projection.", "post_projection.")):
                    yield f"model.{name}", w
                elif "lm_head" in name:
                    yield name, w
                # masked_embedding.* intentionally dropped in v1 (centroid head deferred).
        return AutoWeightsLoader(self).load_weights(remap(weights))
