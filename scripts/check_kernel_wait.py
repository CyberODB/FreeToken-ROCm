"""ROCm Windows fork: check the CPU-MoE kernel spin-wait handshake (used when stream memops fail the probe).
Builds a small executor, compares eager and HIP-graph decode against a CPU reference with new data on
every replay, and times both."""
import sys
import time
from pathlib import Path

import torch

import torch.nn.functional as Fn
from types import SimpleNamespace


# copied from tests/moe/test_cpu_moe.py (that module needs pytest)
def _make_cache(L, E, H, I, scale=0.1):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    gate_up = alloc_pinned_tensor(L * E, 2 * I, H, dtype=torch.bfloat16)
    down = alloc_pinned_tensor(L * E, H, I, dtype=torch.bfloat16)
    gate_up.copy_(torch.randn(L * E, 2 * I, H) * scale)
    down.copy_(torch.randn(L * E, H, I) * scale)
    return SimpleNamespace(
        quant_format="bf16",
        bank_sources={"gate_up": list(gate_up.split(E)), "down": list(down.split(E))},
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


def _reference(cache, layer, hidden, ids, w, act="silu", apply_in=False):
    gu = cache.bank_sources["gate_up"][layer].float()
    dn = cache.bank_sources["down"][layer].float()
    I = gu.shape[1] // 2
    H = gu.shape[2]
    bs, top_k = ids.shape
    xf = hidden.float().cpu()
    out = torch.zeros(bs, H, dtype=torch.float32)
    for t in range(bs):
        for k in range(top_k):
            e = int(ids[t, k])
            if e < 0:
                continue
            wi = float(w[t, k]) if apply_in else 1.0
            wo = 1.0 if apply_in else float(w[t, k])
            gate = (gu[e, :I, :] @ xf[t]) * wi
            up = (gu[e, I : 2 * I, :] @ xf[t]) * wi
            g = (Fn.silu(gate) * up).bfloat16().float()
            out[t] += wo * (dn[e] @ g)
    return out



from freetoken.moe.cpu_executor import CpuMoeExecutor  # noqa: E402

torch.manual_seed(0)
L, E, H, I, top_k, bs = 4, 16, 1024, 512, 4, 1
dev = torch.device("cuda")
cache = _make_cache(L, E, H, I)
ex = CpuMoeExecutor(cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
                    num_threads=0, max_tokens=bs, device=dev)
print(f"flag_sync={ex._flag_sync} kernel_io={ex._kernel_io} kernel_wait={getattr(ex, '_kernel_wait', None)} "
      f"graph_safe(bf16)={ex.graph_capture_safe(torch.bfloat16)}")

hidden = torch.empty(bs, H, device=dev, dtype=torch.bfloat16)
ids = torch.empty(bs, top_k, device=dev, dtype=torch.int32)
w = torch.empty(bs, top_k, device=dev, dtype=torch.float32)


def fill(seed):
    g = torch.Generator().manual_seed(seed)
    hidden.copy_(torch.randn(bs, H, generator=g))
    ids.copy_(torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(bs)]).int())
    w.copy_(torch.rand(bs, top_k, generator=g))


def err(out, layer):
    ref = _reference(cache, layer, hidden, ids.cpu(), w.cpu())
    return float((out.float().cpu() - ref).abs().max() / (ref.abs().max() + 1e-6))


worst = 0.0
for i in range(5):
    fill(i)
    worst = max(worst, err(ex.decode(i % L, hidden, w, ids), i % L))
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(200):
    ex.decode(i % L, hidden, w, ids)
torch.cuda.synchronize()
eager_ms = (time.perf_counter() - t0) / 200 * 1e3
print(f"eager: max rel err {worst:.4f}, {eager_ms:.3f} ms per layer step")

if not ex.graph_capture_safe(torch.bfloat16):
    print("graph not safe: stop")
    sys.exit(1)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    outs = [ex.decode(layer, hidden, w, ids) for layer in range(L)]
worst = 0.0
for i in range(10):
    fill(100 + i)
    graph.replay()
    torch.cuda.synchronize()
    worst = max(worst, max(err(outs[layer], layer) for layer in range(L)))
t0 = time.perf_counter()
for i in range(100):
    graph.replay()
torch.cuda.synchronize()
graph_ms = (time.perf_counter() - t0) / (100 * L) * 1e3
print(f"graph: max rel err {worst:.4f} over 10 replays with new data, {graph_ms:.3f} ms per layer step")
print("PASS" if worst < 2e-2 else "FAIL")
