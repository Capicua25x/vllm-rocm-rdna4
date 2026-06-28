# Gemma-4 26B-A4B NVFP4 on RDNA4 — gap analysis (2026-06-22, Opus 4.8)

## Status
- **GAP 1 — ROUTING: DONE + PROVEN.** `rdna_nvfp4_moe.patched.py` adds `RoutingMethodType.Custom` to the
  allow-list + a Custom branch in apply() (softmax-all→top-k→renorm). CPU-proven == Gemma reference
  (`routing_equiv_check.py`). GPU-confirmed: cleared the exact `routing method 6` gate that killed the
  unpatched 26B. This is correct and necessary.
- **GAP 2 — input_global_scale: DONE.** `gemma4.patched.py` skips `experts.*.input_global_scale` (RDNA
  kernel is weight-only — apply() asserts a1q_scale is None — so the W4A4 activation scale is unused).
  Mirrors the existing `.bias` skip. GPU-confirmed: that KeyError gone.
- **GAP 3 — weight_global_scale: OPEN (the real work).** `KeyError: experts.0.down_proj.weight_global_scale`.

## Root cause of GAP 3 (traced, not guessed)
The destination params DO exist — `CompressedTensorsW4A4Nvfp4MoEMethod.create_weights`
(compressed_tensors_moe.py ~L394) registers per-expert:
`w13_weight_packed/w2_weight_packed`, `w13_weight_scale/w2_weight_scale`,
`w13_weight_global_scale/w2_weight_global_scale` (L472/481), `w13_input_global_scale/w2_input_global_scale` (L492/501).

The bug is in **gemma4.py MoE weight LOADING**, which only routes the main weight:
- `expert_params_mapping` (~L953) maps `experts.{id}.{gate,down,up}_proj` → `experts.w13_weight`/`experts.w2_weight`
  ONLY — no entries for the scale suffixes.
- The expert loop (~L1018) does `moe_name = name.replace(weight_name, param_name)`; for a scale tensor the
  resulting name (`experts.w13_weight.weight_global_scale`) isn't a real param → `moe_name not in params_dict`
  → `continue` → falls through to direct load → KeyError.
- Worse, the loop **hardcodes** the sub-name: `weight_loader(param, loaded_weight, weight_name + ".weight", ...)`
  (~L1039). So even with mapping entries added, the loader is always told ".weight".
- `_weight_iterator` (~L1186) is built for the **fused 3D HF layout** (`experts.gate_up_proj [E,2I,H]` →
  explode). RedHatAI's checkpoint is the **per-expert separate quantized layout**
  (`experts.{id}.{proj}.{weight_packed,weight_scale,weight_global_scale,input_global_scale}`). Only the
  weight+weight_scale of that path is wired; the NVFP4 global scales are not.

## The fix (approach)
Rewire gemma4.py's MoE expert loading to dispatch EACH NVFP4 suffix to its registered param:
- `weight_packed`  → `experts.w{13,2}_weight`        (sub-name ".weight" / ".weight_packed")
- `weight_scale`   → `experts.w{13,2}_weight_scale`  (sub-name ".weight_scale")
- `weight_global_scale` → `experts.w{13,2}_weight_global_scale` (".weight_scale_2"/global)
- `input_global_scale`  → skip (weight-only kernel)
Cleanest: replace the hand-built mapping + hardcoded ".weight" with vLLM's
`FusedMoE.make_expert_params_mapping(...)` IF it emits the NVFP4 scale entries in this build; else extend the
hand-built mapping per-suffix and pass the matching sub-name to the weight_loader. Verify the FusedMoE
weight_loader's expected suffix tokens for nvfp4 (grep its `weight_loader` / `_load_per_tensor_weight_scale`).

## Why this needs a LIVE GPU window (not blind staging)
Unlike routing (CPU-provable), this can't be validated offline — it needs the real FusedMoE layer + weight_loader,
which only exist at model construction. Weight-loading bring-up reveals errors ONE AT A TIME as the loader runs;
expect 1-3 more suffix/shape gaps after weight_global_scale. Efficient path = a focused ~30-60 min window
iterating: patch → `launch-26b-patched.sh` → read next error → repeat. ~1 min/reload.

## Test loop
1. stop your vLLM service  2. `launch-26b-patched.sh` (dual bind-mount)  3. read /tmp log for next KeyError/shape error
4. patch gemma4.patched.py  5. relaunch. When it serves: prompt it; coherent output = full MoE path works
(routing already proven). Then restart your vLLM service.
