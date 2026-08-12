import torch
torch.manual_seed(0)
# Gemma's reference routing (gemma4.Gemma4MoE.routing_function): softmax over ALL experts -> top-k -> renorm
def gemma_reference(logits_fp32, k):
    probs = torch.softmax(logits_fp32, dim=-1)
    topv, topi = torch.topk(probs, k, dim=-1)
    w = topv / topv.sum(dim=-1, keepdim=True)
    return w, topi.to(torch.int64)
# EXACT code from the patched apply() Custom branch
def patched_branch(router_logits, k, out_dtype):
    probs = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(out_dtype)
    return topk_weights, topk_ids.to(torch.int32)
print(f"{'dtype':8} {'E':>4} {'k':>3} | {'ids_exact':>9} {'max_w_diff':>11} {'sum~1_err':>10}")
ok=True
for dtype in (torch.float32, torch.bfloat16):
    for E,k in ((8,2),(16,2),(32,4),(64,6),(128,8)):
        T=8192
        logits=torch.randn(T,E,dtype=dtype)
        rw,ri=gemma_reference(logits.float(),k)
        pw,pi=patched_branch(logits,k,dtype)
        ids_exact=torch.equal(pi.to(torch.int64),ri)
        wdiff=(pw.float()-rw).abs().max().item()
        sumerr=(pw.float().sum(-1)-1).abs().max().item()
        ok &= ids_exact and wdiff < (3e-2 if dtype==torch.bfloat16 else 1e-5)
        print(f"{str(dtype).split('.')[-1]:8} {E:>4} {k:>3} | {str(ids_exact):>9} {wdiff:>11.2e} {sumerr:>10.2e}")
print("\nRESULT:", "PASS — patched branch == Gemma reference routing" if ok else "FAIL")
