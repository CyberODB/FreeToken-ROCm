"""Isolated repro of the GGUF MoE expert GEMM (ggml mul_mat_id: MMVQ for small T, MMQ above) on this GPU.

Why: on the RX 9070 (gfx1201) the 35B-A3B IQ4_NL model dies in warmup prefill with "unspecified launch
failure in ggml_cuda_mul_mat_q" (2026-11-03). This runs that kernel alone on random packed weights at the
model's real shapes, with no model load and no host-RAM pinning, and checks results against gguf-py's CPU
dequantization.

Each case runs in a child process; a GPU fault kills the child, the parent records FAIL and (with
--keep-going) restarts a child for the remaining cases. Results go to the console and to logs/.

Usage (from scripts\\ggml-moe-repro.cmd, which sets the ROCm env):
    ggml-moe-repro                          default sweep, stops at the first failure
    ggml-moe-repro --keep-going             run every case
    ggml-moe-repro --types IQ4_NL --tokens 80 --routes same     one case family
    ggml-moe-repro --list                   print the cases and exit
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time

# Qwen3.6-35B-A3B: hidden 2048, expert intermediate 512, 256 experts, top-8.
SHAPES = {"gate_up": (1024, 2048), "down": (2048, 512)}  # (N rows, K) per expert
NUM_EXPERTS, TOP_K = 256, 8
# byte offsets of the fp16 scale fields inside one block, set to small sane values so outputs stay finite
F16_FIELDS = {"Q4_0": [0], "Q8_0": [0], "IQ4_NL": [0], "IQ4_XS": [0], "IQ3_S": [0], "IQ2_S": [0],
              "Q4_K": [0, 2], "Q5_K": [0, 2], "Q3_K": [108], "Q6_K": [208]}
DEFAULT_TYPES = "IQ4_NL,Q4_0,Q4_K,Q6_K,IQ3_S"
DEFAULT_TOKENS = "1,8,9,16,80,128,512"  # 80 and 128 are the engine's warmup prefill lengths
CASE_TIMEOUT_S = 180


def cases_from(args) -> list[dict]:
    out = []
    for ty, shape, t, route, dt in itertools.product(
        args.types.split(","), args.shapes.split(","), [int(x) for x in args.tokens.split(",")],
        args.routes.split(","), args.dtypes.split(","),
    ):
        out.append({"type": ty, "shape": shape, "tokens": t, "route": route, "dtype": dt})
    return out


def name(c: dict) -> str:
    return f"{c['type']:<6} {c['shape']:<7} T={c['tokens']:<4} route={c['route']:<6} {c['dtype']}"


# ---------------------------------------------------------------- child: runs cases, one line per result
def child(cases: list[dict]) -> None:
    import numpy as np
    import torch

    from freetoken.kernel import ggml
    from freetoken.layers.quantization.ggml_types import pad_rows
    from freetoken.third_party.gguf import GGML_QUANT_SIZES, GGMLQuantizationType as QType
    from freetoken.third_party.gguf import quants

    print(f"READY {torch.cuda.get_device_name(0)}", flush=True)
    weights: dict = {}

    def get_weights(ty: str, shape: str):
        key = (ty, shape)
        if key in weights:
            return weights[key]
        weights.clear()  # one weight set on the GPU at a time
        torch.cuda.empty_cache()
        q = QType[ty]
        block, bbytes = GGML_QUANT_SIZES[q]
        n, k = SHAPES[shape]
        rng = np.random.default_rng(1103)
        raw = rng.integers(0, 256, (NUM_EXPERTS, n, k // block, bbytes), dtype=np.uint8)
        for off in F16_FIELDS[ty]:
            d = rng.uniform(1e-3, 4e-3, raw.shape[:-1]).astype(np.float16).view(np.uint8).reshape(*raw.shape[:-1], 2)
            raw[..., off:off + 2] = d
        packed = raw.reshape(NUM_EXPERTS, n, (k // block) * bbytes)
        pad = pad_rows(int(q), k)
        storage = torch.zeros(NUM_EXPERTS, n + pad, packed.shape[-1], dtype=torch.uint8, device="cuda")
        storage[:, :n].copy_(torch.from_numpy(packed))
        weights[key] = (storage[:, :n], packed, q)
        return weights[key]

    def activation(t: int, k: int, dtype):
        rng = np.random.default_rng(19)
        half = rng.integers(-127, 128, (t, k // 32, 16))
        half[..., 0] = 127  # exactly representable in q8_1, zero block sums: isolates the weight math
        v = np.concatenate([half, -half], axis=-1).reshape(t, k).astype(np.float32) / 64
        return torch.from_numpy(v).to("cuda", dtype)

    for i, c in enumerate(cases):
        w, packed, q = get_weights(c["type"], c["shape"])
        n, k = SHAPES[c["shape"]]
        t = c["tokens"]
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[c["dtype"]]
        # gate_up shares one [T, K] input across routes; down gets the per-route glu output [T, k, K] (gguf.py apply)
        x = activation(t, k, dtype) if c["shape"] == "gate_up" else activation(t * TOP_K, k, dtype).view(t, TOP_K, k)
        rng = np.random.default_rng(7 + t)
        if c["route"] == "same":  # warmup prefill: all-zero input ids, so every token picks the same experts
            row = rng.choice(NUM_EXPERTS, TOP_K, replace=False)
            ids_np = np.tile(row, (t, 1))
        else:
            ids_np = np.stack([rng.choice(NUM_EXPERTS, TOP_K, replace=False) for _ in range(t)])
        ids = torch.from_numpy(ids_np.astype(np.int32)).cuda()
        kernel = "MMVQ" if t <= ggml.mmvq_max_tokens(int(q)) else "MMQ"
        print(f"START {i} {kernel}", flush=True)
        t0 = time.perf_counter()
        out = ggml.mul_mat_id(w, int(q), x, ids)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        # check up to 48 (token, route) pairs against CPU dequantization
        got = out.float().cpu().numpy()
        xs = x.float().cpu().numpy()
        pairs = [(a, b) for a in range(t) for b in range(TOP_K)]
        pairs = [pairs[j] for j in np.random.default_rng(3).choice(len(pairs), min(48, len(pairs)), replace=False)]
        deq: dict = {}
        worst = 0.0
        finite = bool(np.isfinite(got).all())
        for a, b in pairs:
            e = int(ids_np[a, b])
            if e not in deq:
                deq[e] = quants.dequantize(packed[e], q).astype(np.float32)
            ref = deq[e] @ (xs[a] if xs.ndim == 2 else xs[a, b])
            err = float(np.abs(got[a, b] - ref).max() / (np.abs(ref).max() + 1e-6))
            worst = max(worst, err)
        tol = 2e-2 if dtype != torch.float32 else 2e-3
        verdict = "OK" if finite and worst <= tol else "WRONG"
        print(f"DONE {i} {kernel} {verdict} relerr={worst:.2e} finite={finite} {ms:.1f}ms", flush=True)
    print("ALLDONE", flush=True)


# ---------------------------------------------------------------- parent: spawns children, reports
def parent(args) -> int:
    cases = cases_from(args)
    if args.list:
        for i, c in enumerate(cases):
            print(f"{i + 1:>3}  {name(c)}")
        return 0
    logdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, f"ggml-moe-repro-{time.strftime('%Y%m%d-%H%M%S')}.txt")
    log = open(logpath, "w", encoding="utf-8")

    def emit(s: str) -> None:
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    emit(f"ggml MoE repro: {len(cases)} cases, shapes {SHAPES}, E={NUM_EXPERTS} top_k={TOP_K}")
    child_env = dict(os.environ)
    if args.no_serialize:  # run kernels async like the app does; a fault may then surface at a later kernel
        child_env.pop("AMD_SERIALIZE_KERNEL", None)
    emit(f"env AMD_SERIALIZE_KERNEL={child_env.get('AMD_SERIALIZE_KERNEL', '')}  log {logpath}")
    results: list[tuple[str, str]] = []
    todo = list(range(len(cases)))
    t_start = time.time()
    fails = 0
    while todo:
        batch = [cases[i] for i in todo]
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child", json.dumps(batch)], env=child_env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding="utf-8", errors="replace")
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [lines.put(l) for l in p.stdout] + [lines.put(None)], daemon=True).start()
        current, tail, ended = None, [], False
        while True:
            try:
                line = lines.get(timeout=CASE_TIMEOUT_S)
            except queue.Empty:
                p.kill()
                tail.append(f"(no output for {CASE_TIMEOUT_S}s: hang, killed)")
                break
            if line is None:
                break
            line = line.rstrip()
            parts = line.split()
            if line.startswith("READY"):
                emit(f"child up: {line[6:]}")
            elif line.startswith("START"):
                current = int(parts[1])
                tail = []
            elif line.startswith("DONE"):
                j = int(parts[1])
                ci = todo[j]
                done = len(results) + 1
                eta = (time.time() - t_start) / done * (len(cases) - done)
                emit(f"[{done}/{len(cases)}] {name(cases[ci])}  {parts[2]:<4} {' '.join(parts[3:])}  ETA {eta:.0f}s")
                results.append((name(cases[ci]), parts[3]))
                if parts[3] != "OK":
                    fails += 1
                current = None
            elif line.startswith("ALLDONE"):
                ended = True
            else:
                tail.append(line)
                tail = tail[-12:]
        p.wait()
        if ended:
            break
        # the child died or hung during case `current` (or before its first case)
        if current is None:
            emit("child failed before running a case:")
            for l in tail:
                emit("    " + l[:300])
            break
        ci = todo[current]
        fails += 1
        results.append((name(cases[ci]), "CRASH"))
        emit(f"[{len(results)}/{len(cases)}] {name(cases[ci])}  CRASH (exit {p.returncode})")
        for l in [l for l in tail if l.strip()][-6:]:
            emit("    " + l[:300])
        todo = todo[current + 1:]
        if not args.keep_going:
            emit("stopped at the first crash (pass --keep-going to run the rest)")
            break
    emit(f"summary: {sum(r == 'OK' for _, r in results)} OK, {fails} not OK, {len(cases) - len(results)} not run, "
         f"{time.time() - t_start:.0f}s")
    for n_, r in results:
        if r != "OK":
            emit(f"  {r:<6} {n_}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", default=DEFAULT_TYPES)
    ap.add_argument("--shapes", default="gate_up,down")
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--routes", default="same,random")
    ap.add_argument("--dtypes", default="bf16")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--no-serialize", action="store_true", help="drop AMD_SERIALIZE_KERNEL for the child (app-like async launches)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        child(json.loads(args.child))
        return 0
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
