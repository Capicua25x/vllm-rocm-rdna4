#!/usr/bin/env python3
# Re-graft the RDNA4 MXFP4 enablement onto the v0.19.1 triton_kernels (3 files whose base differs from the
# earlier extract). Runs as root inside the serve container. Idempotent-ish: checks before patching.
import sys, py_compile
S = "/build/vllm/vllm"
TK = f"{S}/third_party/triton_kernels"

def patch(path, edits, must_contain=None):
    with open(path) as f: s = f.read()
    if must_contain and must_contain in s:
        print(f"  SKIP {path} (already grafted)"); return
    for old, new in edits:
        if old not in s:
            print(f"  !! anchor NOT found in {path}:\n     {old[:80]!r}"); sys.exit(3)
        s = s.replace(old, new, 1)
    with open(path, "w") as f: f.write(s)
    py_compile.compile(path, doraise=True)
    print(f"  grafted + compiles: {path}")

# ---- target_info.py: add RDNA helpers ----
patch(f"{TK}/target_info.py", [
 ('    "get_cdna_version",\n',
  '    "get_cdna_version",\n    "get_rdna_version",\n    "get_rdna_version_host",\n'),
 ('    "is_hip_cdna4",\n',
  '    "is_hip_cdna4",\n    "is_hip_rdna4",\n'),
 ('@triton.constexpr_function\ndef has_tma_gather():',
  '''@triton.constexpr_function
def get_rdna_version():
    target = tl.target_info.current_target()
    if target.backend != 'hip':
        return -1
    arch = target.arch
    if arch[:5] == 'gfx11':
        return 3
    if arch[:5] == 'gfx12' and arch[:7] != 'gfx1250':
        return 4
    return -1


def get_rdna_version_host() -> int:
    if not torch.cuda.is_available():
        return -1
    gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
    if not gcn_arch:
        return -1
    arch = gcn_arch.split(":")[0]
    if arch[:5] == 'gfx11':
        return 3
    if arch[:5] == 'gfx12' and arch[:7] != 'gfx1250':
        return 4
    return -1


@triton.constexpr_function
def is_hip_rdna4():
    return get_rdna_version() == 4


@triton.constexpr_function
def has_tma_gather():'''),
], must_contain="def get_rdna_version_host")

# ---- opt_flags.py: import + RDNA4 tile branch ----
patch(f"{TK}/matmul_ogs_details/opt_flags.py", [
 ('from triton_kernels.target_info import get_cdna_version',
  'from triton_kernels.target_info import get_cdna_version, get_rdna_version_host'),
 ('    is_cdna4 = get_cdna_version() == 4',
  '''    if get_rdna_version_host() == 4:
        block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 64))
        if constraints.get("block_m", None):
            block_m = constraints["block_m"]
        ret = OptFlags(
            block_m=block_m, block_n=128, block_k=64,
            num_warps=4, num_stages=2, group_m=4,
            xcd_swizzle=1, w_cache_modifier=None,
            split_k=1, is_persistent=False, idle_sms=0,
            epilogue_subtile=constraints.get('epilogue_subtile', 1),
            arch=None,
            target_kernel_kwargs={"waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 1},
        )
        all_constraints_satisfied(ret, constraints)
        return ret

    is_cdna4 = get_cdna_version() == 4'''),
], must_contain="get_rdna_version_host() == 4")

# ---- _matmul_ogs.py: import + RDNA_VALUE static_assert/divisor/dispatch ----
patch(f"{TK}/matmul_ogs_details/_matmul_ogs.py", [
 ('from triton_kernels.tensor_details.layout_details.cdna4_scale import unswizzle_mx_scale_cdna4',
  'from triton_kernels.tensor_details.layout_details.cdna4_scale import unswizzle_mx_scale_cdna4\nfrom triton_kernels.tensor_details.layout_details.rdna_value import mxfp4_dequant_rdna'),
 ('tl.static_assert(SWIZZLE_MX_VALUE == "HOPPER_VALUE" or SWIZZLE_MX_VALUE is None, "Only Hopper swizzling is supported for values")',
  'tl.static_assert(SWIZZLE_MX_VALUE == "HOPPER_VALUE" or SWIZZLE_MX_VALUE == "RDNA_VALUE" or SWIZZLE_MX_VALUE is None, "Only Hopper and RDNA swizzling is supported for values")'),
 ('''            W_K_DIVISOR: tl.constexpr = 1
            W_K_MULTIPLIER: tl.constexpr = 2
            W_N_DIVISOR: tl.constexpr = 4
        else:''',
  '''            W_K_DIVISOR: tl.constexpr = 1
            W_K_MULTIPLIER: tl.constexpr = 2
            W_N_DIVISOR: tl.constexpr = 4
        elif SWIZZLE_MX_VALUE == "RDNA_VALUE":
            tl.static_assert(is_mxfp4, "Only mxfp4 is supported for RDNA dequant")
            tl.static_assert(not is_x_microscaled)
            W_K_DIVISOR: tl.constexpr = 2
            W_K_MULTIPLIER: tl.constexpr = 1
            W_N_DIVISOR: tl.constexpr = 1
        else:'''),
 ('''                acc = tl.dot(w, x, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
                acc = acc.trans()
            else:
                rhs_k_pack: tl.constexpr = W_TRANSPOSE or not is_w_microscaled or W_K_DIVISOR != 2''',
  '''                acc = tl.dot(w, x, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
                acc = acc.trans()
            elif SWIZZLE_MX_VALUE == "RDNA_VALUE":
                rdna_out_type: tl.constexpr = x.dtype
                w_dequant = mxfp4_dequant_rdna(w, w_scales, mx_axis=1, OUT_DTYPE=rdna_out_type)
                acc = tl.dot(x, w_dequant, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
            else:
                rhs_k_pack: tl.constexpr = W_TRANSPOSE or not is_w_microscaled or W_K_DIVISOR != 2'''),
], must_contain='SWIZZLE_MX_VALUE == "RDNA_VALUE"')

print("OVERLAY GRAFT OK")
