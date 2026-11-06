"""ROCm Windows fork: time GGUF expert bank loading, mmap copy vs threaded file reads, on separate layer ranges
(so neither run reads pages the other just cached), and check both give the same bytes.
usage: check_gguf_load.py <model.gguf> [layers_per_run=8] [threads=8]
Loads the worktree's experts.py next to the installed freetoken package."""
import gc
import importlib.util
import sys
import time
from pathlib import Path

import torch

import freetoken.models.gguf  # noqa: F401  (parent package for the relative imports)

src = Path(__file__).resolve().parents[1] / "python" / "freetoken" / "models" / "gguf" / "experts.py"
spec = importlib.util.spec_from_file_location("freetoken.models.gguf.experts_wt", src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 8
torch.cuda.init()


def load(first, layers, t):
    mod._READ_THREADS = t
    t0 = time.perf_counter()
    banks = mod.load_gguf_expert_banks(path, layers, first_layer=first)
    dt = time.perf_counter() - t0
    gib = sum(x.numel() * x.element_size() for ts in banks.sources.values() for x in ts) / 2**30
    return banks, dt, gib


a, _, _ = load(0, 1, 0)
b, _, _ = load(0, 1, threads)
same = all(torch.equal(x, y) for role in a.sources for x, y in zip(a.sources[role], b.sources[role]))
print(f"layer 0 bytes identical: {same}", flush=True)
del a, b
gc.collect()

for first, t in ((1, 0), (1 + n, threads), (1 + 2 * n, threads), (1 + 3 * n, 0)):
    banks, dt, gib = load(first, n, t)
    print(f"layers {first}..{first + n - 1} {'threaded x' + str(t) if t else 'mmap copy':12s}: {gib:.2f} GiB in {dt:.1f} s = {gib * 1024 / dt:.0f} MiB/s", flush=True)
    del banks
    gc.collect()
print("PASS" if same else "FAIL")
