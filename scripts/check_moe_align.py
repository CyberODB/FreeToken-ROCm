"""Check moe_align_block_size on this GPU against its contract (upstream #687: Triton 3.8 on ROCm
silently puts tokens in the wrong expert blocks). Prints one line per case and PASS/FAIL."""
import torch

from freetoken.moe.fused import moe_align_block_size


def check(t, k, e, bs, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(e, generator=g)[:k] for _ in range(t)]).to(torch.int32).cuda()
    sorted_ids, expert_ids, npp = moe_align_block_size(ids, bs, e)
    torch.cuda.synchronize()
    flat = ids.flatten().cpu()
    n = flat.numel()
    s, ex, npp = sorted_ids.cpu(), expert_ids.cpu(), int(npp.item())
    want = sum(-(-int((flat == x).sum()) // bs) * bs for x in range(e))
    errs = []
    if npp != want:
        errs.append(f"num_post_padded {npp} != {want}")
    seen = torch.zeros(n, dtype=torch.int32)
    for b in range(npp // bs):
        x = int(ex[b])
        for r in s[b * bs:(b + 1) * bs].tolist():
            if r < n:
                seen[r] += 1
                if int(flat[r]) != x:
                    errs.append(f"block {b} expert {x} holds route {r} of expert {int(flat[r])}")
    if not bool((seen == 1).all()):
        errs.append(f"{int((seen != 1).sum())} routes missing or duplicated")
    print(f"T={t} k={k} E={e} bs={bs}: {'PASS' if not errs else 'FAIL ' + '; '.join(errs[:3])}")
    return not errs


import triton

print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name())
ok = all(check(t, 8, 256, bs, i) for i, (t, bs) in enumerate([(1, 16), (4, 16), (37, 16), (512, 64), (2048, 64)]))
print("ALL PASS" if ok else "SOME FAILED")
