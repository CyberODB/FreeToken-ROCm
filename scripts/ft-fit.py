"""Fit a GGUF to this PC and recommend a launch config (a model card). Plan: T:\\robot\\docs\\19-freetoken-fit-flow-plan.md

  py scripts/ft-fit.py <model.gguf> --plan-only        size check + upgrades, no GPU
  py scripts/ft-fit.py <model.gguf> [--max-runs 10]    guarded run ladder, then speedprobe/cards/<model>.md

Stages: 1 inventory/size check, 2 upgrades (MTP), 3 strategy, 4 memory ratio, 4s expert slots against context
(2026-11-11; --no-slots skips it; undo: ft-fit.py.bak-20261111-slots), 5 spec decoding, 6 flags,
7 fit confirmation with vision + a 5k prompt, 8 context, then the card. Every GPU run is gpu-guard.py around ft-speedprobe.py.
2026-11-12 (undo: ft-fit.py.bak-20261112-stages):
  4s  --slots-mode auto: one --moe-cache-auto run (KV floor --kv-reserve) instead of the sweep; the engine's slot count
      is read from its log and becomes the card's cache.
  5   --dflash-head <path>: a DFlash draft is tried after MTP (k 3/5/7/11/15 up to block_size-1, 2026-11-13); the faster of the two is the spec config.
  6   a fit-flags.json entry may carry "values" (one run each), "tests", "metric" (decode | prefill:<test> | total:<test>)
      and "lossless"; kept only when lossless is true, the text is identical, the metric gains over 2% and nothing else
      in the run is over 5% worse. A tests string other than "decode" gets its own plain reference run.
  6p  parallel requests (2, 4, 8 at once with graphs at N; 4 at graph bs 1 = live today; FT_FIT_PAR_EXTRA adds the args to 2 and 4; test cN): report only, never changes the winner; --no-parallel skips it.
2026-11-13 flow rework (undo: ft-fit.py.bak-20261113-flow). The order is now:
  3+4 strategy/ratio -> 5 spec (MTP, DFlash, then --token-map on the MTP winner) -> 6 slots on the config in use, against
  --ctx-floor (fastest count that holds it) -> 7 final: the launch line as stamped, in one run, with the class's tune flags
  -> 8 context fill on that run -> 8-kv8 one 8-bit KV row (report only; --no-kv8).
  Flags and the parallel report moved to --tune: once per model class, saved to speedprobe/tune/<class>.json, flag winners
  run twice. A normal fit reads that file and only checks the kept flags in its final run.
2026-11-14 (undo: ft-fit.py.bak-20261114-fitflow): 4-squeeze when no hybrid ratio loads (mr1, then 3000 slots, then naive
  cache; first that loads wins); 8-bit KV only as a rescue when 16-bit misses the floor (lossy, own launch line);
  winner calls under 5% rerun both sides; dense class default --max-prefill-length 2048; the card's "What failed" section.
  Later the same day (undo: ft-fit.py.bak-20261114-slotroom): a hybrid run whose 5000 expert slots do not fit is retried once
  at the room the engine reports, and the squeeze never asks for more than that room.
  Later still (undo: ft-fit.py.bak-20261114-goal): the context is a goal (--ctx-goal, 64K) over a floor (--ctx-floor, now
  32K, was the goal): slots aim at the goal; met / under goal (usable) / failed (under the floor); 8-bit KV rescues only
  under the floor.
  Pre-flight (same undo): before any load, the model's arch and tensor types are checked against the engine's own tables
  (stop with a reason, --no-preflight overrides) and every fixed slot count is capped at the model's expert total.
Pass/fail comes from the probe's results.jsonl row (its exit code is 0 on a failed load), kills from the
guard journal. Results go to speedprobe/fit/<model>.jsonl; a rerun reuses rows with the same name and args.
Stdlib only, never imports torch.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_header import read_header, type_name  # noqa: E402
from ft_daemon_env import daemon_env  # noqa: E402

# 2026-11-14: a Windows GPU reset (TDR) kills a run from outside; the next load straight after can trip another one.
# Undo for this block and its uses: restore ft-fit.py.bak-20261114-benchsafe.
GPU_RESET_PAT = re.compile(r"hipError|unspecified launch|device lost|DEVICE_LOST|HIP abort|ROCm error", re.I)
OOM_PAT = re.compile(r"out of memory|OutOfMemory", re.I)
WER_QUEUE = r"C:\ProgramData\Microsoft\Windows\WER\ReportQueue"
GPU_RESET_SETTLE_S, GPU_RESET_STOP = 20, 2
RAM_SETTLE_S = 30  # after the guard stops a run on host RAM, before the next load (the bench waits the same)
RESETS = {"n": 0}  # GPU resets seen by this fit, for write_out


def gpu_reset_since(t0: float, log_path: str) -> str | None:
    """Why this looks like a Windows GPU reset since t0 (a LiveKernelEvent 141/193 report, or a HIP device error in the
    run's own log, which the probe writes fresh each run), else None."""
    try:
        for d in os.listdir(WER_QUEUE):
            st = os.stat(os.path.join(WER_QUEUE, d))  # creation time: WER touches the folder again later (13:59 report, 18:14 mtime)
            if re.search(r"Kernel_(141|193)", d) and getattr(st, "st_birthtime", st.st_ctime) >= t0 - 5:
                return "LiveKernelEvent " + re.search(r"Kernel_(141|193)", d).group(1)
    except OSError:
        pass
    try:
        if os.path.getmtime(log_path) < t0:  # a probe that never started leaves the last run's log: not this run's evidence
            return None
        for line in open(log_path, encoding="utf-8", errors="replace").read()[-65536:].splitlines():
            m = GPU_RESET_PAT.search(line)
            if m and not OOM_PAT.search(line):  # an out-of-memory error is a config result, not a reset
                return m.group(0)
    except OSError:
        pass
    return None

GIB = 1 << 30
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_OUT = os.path.join(REPO, "speedprobe")
RESULTS = os.path.join(PROBE_OUT, "results.jsonl")
GUARD_DIR = os.environ.get("FT_GUARD_DIR", r"T:\robot\state\gpu-guard")
JOURNAL = os.path.join(GUARD_DIR, "journal.jsonl")
SIDECAR_DIRS = [r"T:\ai-models\gguf\local"]
FLAGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit-flags.json")
RATIOS = (0.80, 0.85, 0.90)
SAME = 0.02  # gains under 2% are noise (doc 18: +-1-2 tok/s per 512-token run)
CLOSE = 0.05  # 2026-11-14: a winner decision with a margin under 5% reruns both sides (repeats spread ~4%) and decides on the means
# 2026-11-14: settings every fit of a class gets, measured, so they don't hang on one tune run. Dense: the 5K prefill went from
# 193.5 to 359.8 tok/s (mean of 2) with a 2048 chunk (qwen38-27b 4.2bpw tune). Dropped like a tune flag if it slows decode.
CLASS_DEFAULTS = {"dense": [{"id": "prefill-chunk-default", "kind": "arg", "name": "--max-prefill-length", "value": "--max-prefill-length 2048",
                             "status": "default", "lossless": True, "keep": True,
                             "result": "class default for dense models (5K prefill 193.5 -> 359.8 tok/s on qwen38-27b, 2026-11-14)"}]}
# 2026-11-11 slot sweep (stage 4s): expert GPU slots traded for KV (context). 5000 is the probe's base; each step down is kept
# while decode stays within SLOT_DECODE and the 5K prefill within SLOT_PREFILL of the 5000 run, and the KV keeps growing.
SLOTS = (4000, 3000, 2000)
SLOT_DECODE, SLOT_PREFILL = 0.05, 0.10
FLAG_WORSE = 0.05  # 2026-11-12 stage 6: a flag is not kept when another number of its run drops more than this
PARALLEL = (2, 4, 8)  # 2026-11-12 stage 6p: requests at once (8 added 2026-11-14)
# 2026-11-13 flow rework (undo: ft-fit.py.bak-20261113-flow)
SLOTS_UP = (6500, 8000)  # tried only when 5000 slots already hold the context floor: more experts on the GPU for speed
TUNE_DIR = os.path.join(PROBE_OUT, "tune")  # <class>.json: the flags and parallel report of a --tune run
MAP_DIR = os.path.join(PROBE_OUT, "token-maps")
ENGINE_PY = os.environ.get("FT_FIT_PYTHON", r"E:\freetoken-rocm\.venv\Scripts\python.exe")  # make_token_map.py needs numpy
KV8 = "fp8_per_token_head"


# ---------- stage 1-2: no GPU ----------

class _MS(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tphys", ctypes.c_ulonglong), ("aphys", ctypes.c_ulonglong),
                ("tpage", ctypes.c_ulonglong), ("apage", ctypes.c_ulonglong), ("tvirt", ctypes.c_ulonglong),
                ("avirt", ctypes.c_ulonglong), ("aext", ctypes.c_ulonglong)]


def ram_gib() -> tuple[float, float]:
    """(total, available) GiB: the same 'available' counter the guard's RAM floor uses."""
    m = _MS(); m.len = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.tphys / GIB, m.aphys / GIB


def find_mmproj(model: str) -> str | None:
    d = os.path.dirname(os.path.abspath(model))
    hits = sorted(glob.glob(os.path.join(d, "mmproj*.gguf")))
    return hits[0] if hits else None


def find_sidecars(arch: str, hidden: int | None) -> list[str]:
    """mtp-*.gguf files that pass mtp.py:check_source's rules for this target (same arch and hidden, only a nextn block)."""
    out = []
    for d in SIDECAR_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "mtp-*.gguf"))):
            try:
                meta, tensors = read_header(p)
            except Exception:
                continue
            blocks = {t["name"].split(".")[1] for t in tensors if t["name"].startswith("blk.")}
            nextn = any(".nextn.eh_proj.weight" in t["name"] for t in tensors)
            a = meta.get("general.architecture")
            h = meta.get(f"{a}.embedding_length")
            if nextn and len(blocks) == 1 and a == arch and (hidden is None or h is None or int(h) == int(hidden)):
                out.append((p, meta.get(f"{a}.expert_count")))
    return out


def inventory(model: str, vram_gib: float, desktop_gib: float, extra_gib: float) -> dict:
    meta, tensors = read_header(model)
    arch = meta.get("general.architecture", "?")
    g = lambda k, d=None: meta.get(f"{arch}.{k}", d)
    exps = [t for t in tensors if "_exps" in t["name"]]
    experts_b = sum(t["nbytes"] for t in exps)
    total_b = sum(t["nbytes"] for t in tensors)
    types: dict[str, int] = {}
    for t in exps:
        if t["name"].endswith(".weight"):
            types[type_name(t["type"])] = types.get(type_name(t["type"]), 0) + 1
    by = {t["name"]: t["type"] for t in tensors}
    # 2026-11-14: the engine's pool room is (layers with expert tensors) x expert_count (pool_policy.py), e.g. gemma4 30 x 128
    moe_layers = {t["name"].split(".")[1] for t in exps if t["name"].startswith("blk.")}
    expert_slots = len(moe_layers) * int(g("expert_count") or 0) or None
    odd = [L for L in range(int(g("block_count", 0) or 0))
           if f"blk.{L}.ffn_gate_exps.weight" in by and by[f"blk.{L}.ffn_gate_exps.weight"] != by.get(f"blk.{L}.ffn_up_exps.weight")]
    # KV per token (bf16 assumed): only full-attention layers hold KV in qwen35*/hybrid archs
    layers, kvh, kl, vl = g("block_count"), g("attention.head_count_kv"), g("attention.key_length"), g("attention.value_length")
    kv_tok = None
    if not (kl and vl) and g("embedding_length") and g("attention.head_count"):  # LFM2 has no key_length: embedding / heads
        kl = vl = int(g("embedding_length")) // int(g("attention.head_count"))
    if layers and kvh and kl and vl and isinstance(kvh, (list, tuple)):  # 2026-11-13: per-layer list (LFM2: 0 on conv layers)
        kv_tok = sum(int(h) for h in kvh) * (int(kl) + int(vl)) * 2
    elif layers and kvh and kl and vl:
        attn_layers = int(layers) // int(g("full_attention_interval", 1) or 1)
        kv_tok = attn_layers * int(kvh) * (int(kl) + int(vl)) * 2
    mmproj = find_mmproj(model)
    mm_b = os.path.getsize(mmproj) if mmproj else 0
    builtin_mtp = any(".nextn.eh_proj.weight" in t["name"] for t in tensors)
    # a sidecar with another expert count is the base model's head on a pruned/finetuned target: it loads, acceptance unknown
    sidecars = [] if builtin_mtp else [p for p, e in find_sidecars(arch, g("embedding_length"))]
    sidecar_exact = bool(sidecars) and any(e == g("expert_count") for _, e in find_sidecars(arch, g("embedding_length")))
    tot_ram, avail = ram_gib()
    usable_vram = vram_gib - desktop_gib
    dense = (total_b - experts_b) / GIB
    host_text = experts_b / GIB + extra_gib
    host_vis = host_text + mm_b / GIB
    kv8k = (kv_tok or 0) * 8192 / GIB
    # fused: everything inside the engine's memory_ratio share of VRAM (a 0.80 run failed on KV with weights at 12.8 of 14.8 GiB)
    if dense + experts_b / GIB + kv8k + mm_b / GIB + 1.0 < vram_gib * 0.9 - desktop_gib:
        verdict = "fused"
    elif dense + kv8k + 1.0 > usable_vram or host_text > tot_ram - 4:
        verdict = "no-fit"
    elif host_text > avail - 2:
        verdict = "tight"  # warn only: the guard stops a run before RAM runs out
    elif host_vis > avail - 2:
        verdict = "tight-vision"  # text fits; vision may not
    else:
        verdict = "hybrid"
    return {
        "model": model, "arch": arch, "layers": layers, "experts": g("expert_count"), "experts_used": g("expert_used_count"),
        "file_gib": round(total_b / GIB, 2), "experts_gib": round(experts_b / GIB, 2), "dense_gib": round(dense, 2),
        "expert_types": types, "gate_up_q8_layers": odd, "kv_bytes_per_token": kv_tok, "expert_slots": expert_slots,
        "tensor_types": sorted({t["type"] for t in tensors}), "sliding_window": g("attention.sliding_window"),
        "kv_gib_8k": round(kv8k, 3), "kv_gib_32k": round((kv_tok or 0) * 32768 / GIB, 3),
        "context_length": g("context_length"), "mmproj": mmproj, "mmproj_gib": round(mm_b / GIB, 2),
        "builtin_mtp": builtin_mtp, "sidecars": sidecars, "sidecar_exact": sidecar_exact, "ram_total_gib": round(tot_ram, 1), "ram_avail_gib": round(avail, 1),
        "vram_usable_gib": round(usable_vram, 2), "host_need_text_gib": round(host_text, 2),
        "host_need_vision_gib": round(host_vis, 2), "verdict": verdict,
    }


def print_inventory(inv: dict) -> None:
    print(f"== {os.path.basename(inv['model'])}  ({inv['arch']}, {inv['layers']} layers, {inv['experts']} experts / {inv['experts_used']} used)")
    print(f"  file {inv['file_gib']} GiB = experts {inv['experts_gib']} + dense {inv['dense_gib']}; expert tensors {inv['expert_types']}")
    if inv["gate_up_q8_layers"]:
        print(f"  gate!=up layers (load as Q8_0): {inv['gate_up_q8_layers']}")
    kv = inv["kv_bytes_per_token"]
    print(f"  KV {kv // 1024 if kv else '?'} KiB/token (bf16 est.): 8k {inv['kv_gib_8k']} GiB, 32k {inv['kv_gib_32k']} GiB; model max ctx {inv['context_length']}")
    if not inv["experts"]:
        print("  dense model: MoE settings are ignored. The size check below leaves out the GDN state pool, MTP head and load overhead,\n"
              "  so 'fused' here does not mean it loads; the ladder squeezes ratio, running requests, cache type and embeddings instead.")
    print(f"  vision: {os.path.basename(inv['mmproj']) + ' ' + str(inv['mmproj_gib']) + ' GiB' if inv['mmproj'] else 'none'}")
    mtp = "built-in nextn block" if inv["builtin_mtp"] else (", ".join(os.path.basename(s) for s in inv["sidecars"]) + ("" if inv["sidecar_exact"] else " (expert count differs: base model's head, acceptance unknown)")
                                                            if inv["sidecars"] else "none (look for an mtp-<model>.gguf; never auto-download)")
    print(f"  MTP draft: {mtp}")
    print(f"  RAM {inv['ram_avail_gib']}/{inv['ram_total_gib']} GiB free; host need text {inv['host_need_text_gib']}, with vision {inv['host_need_vision_gib']} GiB; VRAM usable {inv['vram_usable_gib']}")
    print(f"  verdict: {inv['verdict']}")


# ---------- GPU runs ----------

def slug(model: str) -> str:
    s = re.sub(r"\.gguf$", "", os.path.basename(model), flags=re.I)
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:48].lower()


def last_row(name: str) -> dict | None:
    row = None
    if os.path.isfile(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                if f'"{name}"' in line:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("name") == name:
                        row = r
    return row


def journal_stats(name: str, since: str) -> dict:
    """min available RAM, max VRAM used, kill reasons of the guard run `name` journaled at or after `since`."""
    st = {"min_ram_gib": None, "max_vram_gib": None, "kills": [], "end": None}
    if not os.path.isfile(JOURNAL):
        return st
    with open(JOURNAL, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("name") != name or e.get("t", "") < since:
                continue
            if e["ev"] == "vram":
                st["min_ram_gib"] = e["ram_gib"] if st["min_ram_gib"] is None else min(st["min_ram_gib"], e["ram_gib"])
                st["max_vram_gib"] = e["used_gib"] if st["max_vram_gib"] is None else max(st["max_vram_gib"], e["used_gib"])
            elif e["ev"] == "kill" and not str(e.get("why", "")).startswith("leftover"):
                st["kills"].append(e.get("why"))
            elif e["ev"] == "end":
                st["end"] = e.get("how")
    return st


class Runner:
    def __init__(self, model: str, max_runs: int, load_timeout: int, tag: str | None = None, ctx: int = 0, fresh: bool = False):
        self.model, self.max_runs, self.load_timeout, self.ctx = model, max_runs, load_timeout, ctx
        self.tag = tag or slug(model)
        os.makedirs(os.path.join(PROBE_OUT, "fit"), exist_ok=True)
        self.state_path = os.path.join(PROBE_OUT, "fit", f"{self.tag}.jsonl")
        self.done: dict[str, dict] = {}
        if os.path.isfile(self.state_path) and not fresh:
            with open(self.state_path, encoding="utf-8") as f:
                for line in f:
                    try:  # 2026-11-14: a line torn by a hard reset is skipped, not fatal
                        r = json.loads(line)
                        if r.get("gpu_reset"):  # 2026-11-14: a run killed by a Windows GPU reset says nothing about its config
                            continue
                        self.done[r["key"]] = r
                    except (ValueError, KeyError, TypeError):
                        continue
        self.base_env: dict = {}  # env on every run (the desktop VRAM reserve)
        # 2026-11-14: the engine env ft-speedprobe adds under ours (the daemon launcher's, see ft_daemon_env.py). It is part of
        # the reuse key, so rows probed at the old fixed pin 14 / graph bs 1 are not reused as if they were the daemon's config.
        self.probe_env = {k: os.environ.get(k, v) for k, v in daemon_env().items()}
        self.gpu_resets = 0  # Windows GPU resets during this fit; GPU_RESET_STOP of them end the fit
        self.runs = 0
        self.reused = 0
        self.log: list[dict] = []

    def run(self, stage: str, label: str, args: list[str], tests: str = "decode", vision: bool = False,
            env: dict | None = None, base_args: bool = True, ctx: int | None = None) -> dict | None:
        """One guarded probe run; None when the run budget is spent."""
        name = f"fit-{self.tag}-{label}"[:90]
        ctx = self.ctx if ctx is None else ctx
        env = {**self.base_env, **(env or {})} or None
        key = json.dumps([name, args, tests, vision, env or {}, base_args] + ([ctx] if ctx else []) + [{"probe_env": self.probe_env}])
        if key in self.done:
            r = self.done[key]
            self.reused += 1
            print(f"  [{stage}] {label}: reused  {fmt(r)}")
            self.log.append(r)
            return r
        if self.gpu_resets >= GPU_RESET_STOP:
            print(f"  [{stage}] {label}: skipped, run budget spent (stopped after {self.gpu_resets} Windows GPU resets)")
            return None
        if self.runs >= self.max_runs:
            print(f"  [{stage}] {label}: skipped, run budget ({self.max_runs}) spent")
            return None
        self.runs += 1
        probe = ["py", "-u", "scripts/ft-speedprobe.py", name, "--model", self.model, "--args", " ".join(args + (["--max-seq-len-override", str(ctx)] if ctx else [])),
                 "--tests", tests, "--load-timeout", str(self.load_timeout)]
        probe += ["--vision"] if vision else []
        probe += [] if base_args else ["--no-base-args"]
        for k, v in (env or {}).items():
            probe += ["--env", f"{k}={v}"]
        cmd = ["py", "scripts/gpu-guard.py", "--name", name, "--timeout", str(self.load_timeout + 900), "--"] + probe
        since = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{stage}] {label}: {' '.join(args)}{' +vision' if vision else ''}{' ' + str(env) if env else ''} ({tests}) ...", flush=True)
        t0 = time.time()
        envp = dict(os.environ, FT_GUARD_DIR=GUARD_DIR)
        p = subprocess.run(cmd, cwd=REPO, env=envp, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.stdout.decode("utf-8", "replace")
        row = last_row(name)
        fresh = row is not None and row.get("when", "") >= since[:16]
        js = journal_stats(name, since)
        r = {"key": key, "stage": stage, "label": label, "name": name, "args": args, "tests": tests, "vision": vision,
             "env": env or {}, "ctx": ctx or None, "when": since, "wall_s": round(time.time() - t0), "guard_rc": p.returncode, **js}
        if fresh and not row.get("error") and all(t in row.get("tests", {}) for t in tests.split(",")):
            r["ok"] = True
            r["load_s"] = row.get("load_s")
            r["tps"] = {t: row["tests"][t].get("decode_tps") for t in row["tests"]}
            r["prefill"] = {t: row["tests"][t].get("prefill_tps") for t in row["tests"]}
            r["text"] = row["tests"].get("decode", {}).get("text")
            per = {t: v["per_req_tps"] for t, v in row["tests"].items() if v.get("per_req_tps") is not None}
            if per:  # 2026-11-12: the cN tests' tok/s of each request (their total is already in tps[cN])
                r["per_req"] = per
        else:
            r["ok"] = False
            err = (row or {}).get("error") if fresh else None
            m = re.findall(r"(?:NotImplementedError|RuntimeError|ValueError|Error)[^\n]{0,200}", out)
            r["error"] = (js["kills"][0] if js["kills"] else None) or err or (m[-1] if m else f"no result row (guard rc {p.returncode})")
        try:  # the KV the engine allocated at load: the real context ceiling of this config
            lg = open(os.path.join(PROBE_OUT, f"{name}.log"), encoding="utf-8", errors="replace").read()
            kv = re.findall(r"Allocating (\d+) tokens for KV cache", lg)
            r["kv_tokens"] = int(kv[-1]) if kv else None
            r["kv_short"] = (not r["ok"]) and bool(re.search(r"not enough memory for kv cache|cache budget too small", lg, re.I))
            # 2026-11-12: engine.py logs "--moe-cache-auto resolved moe_cache_size=N num_pages=P (prefill_overlap=..)"
            sa = re.findall(r"moe-cache-auto resolved moe_cache_size=(\d+)", lg)
            r["slots_auto"] = int(sa[-1]) if sa else None
            r["slots_room"] = None if r["ok"] else slots_room_in(lg)
        except OSError:
            r["kv_tokens"] = None
            r["kv_short"] = False
            r["slots_auto"] = None
            r["slots_room"] = None
        reset = gpu_reset_since(t0, os.path.join(PROBE_OUT, f"{name}.log"))  # ok runs too: the 11/14 13:59 reset hit a run that exited rc 0
        if reset:  # 2026-11-14: Windows reset the GPU under this run: its numbers are void; wait, don't cache it, and stop after repeats
            r["gpu_reset"] = True
            r["error"] = f"Windows reset the GPU ({reset}): {r.get('error') or 'numbers discarded'}"
            r["ok"] = False
            self.gpu_resets += 1
            RESETS["n"] = self.gpu_resets
        with open(self.state_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        print(f"      -> {fmt(r)}")
        if reset:
            if self.gpu_resets >= GPU_RESET_STOP:  # run() skips every later run ("run budget spent" in the notes)
                print(f"      !! {self.gpu_resets} GPU resets in this fit: stopping it. The PC needs a look before more GPU runs.", flush=True)
            else:
                print(f"      .. GPU reset: waiting {GPU_RESET_SETTLE_S} s before the next run", flush=True)
                time.sleep(GPU_RESET_SETTLE_S)
        else:
            self.done[key] = r
        self.log.append(r)
        return r


def err(r: dict) -> str:
    """A failed row's error plus a plain-language 'try' nudge from freetoken/load_hints.py (loaded by path: stdlib only)."""
    e = str(r.get("error"))
    try:
        import importlib.util
        sp = importlib.util.spec_from_file_location("load_hints", os.path.join(REPO, "python", "freetoken", "load_hints.py"))
        m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
        tip = m.explain(e)
    except Exception:
        tip = None
    return f"{e} -> {tip}" if tip else e


def fmt(r: dict) -> str:
    mem = f"minRAM {r.get('min_ram_gib')} GiB, maxVRAM {r.get('max_vram_gib')} GiB"
    if r.get("ok"):
        return f"{', '.join(f'{t} {v} tok/s' for t, v in r['tps'].items())}; {mem}"
    return f"FAIL: {err(r)[:420]}; {mem}"


def tps(r: dict | None) -> float:
    return float(r["tps"]["decode"]) if r and r.get("ok") else 0.0


def better(new: dict | None, old: dict | None) -> bool:
    """new beats old only by more than the noise band."""
    return tps(new) > tps(old) * (1 + SAME)


def decide(R: "Runner", new: dict | None, old: dict | None, notes: list[str], what: str) -> bool:
    """better(), except a close call (both ok, margin under CLOSE): both configs run once more and the means decide."""
    if not (new and new.get("ok") and old and old.get("ok")):
        return better(new, old)
    a, b = tps(new), tps(old)
    if abs(a - b) >= CLOSE * max(a, b):
        return better(new, old)
    R.max_runs += 2
    means = []
    for r in (new, old):
        again = R.run(r["stage"], r["label"] + "-again", r["args"], tests=r["tests"], vision=r["vision"], env=r.get("env") or None,
                      base_args=json.loads(r["key"])[5], ctx=r.get("ctx"))
        means.append(round((tps(r) + tps(again)) / 2, 2) if again and again.get("ok") else ("skipped" if again is None else None))
    if "skipped" in means or None in means:
        why = "a rerun was not run (run budget spent)" if "skipped" in means else "a rerun failed"
        notes.append(f"close call, {what}: {a} vs {b} tok/s, {why}; decided on the first runs")
        return better(new, old)
    win = means[0] > means[1] * (1 + SAME)
    notes.append(f"close call, {what}: {a} vs {b} tok/s, so both ran twice; means {means[0]} vs {means[1]}: {'new' if win else 'old'} kept")
    return win


# ---------- ladder ----------

ENGINE_SRC = os.path.join(REPO, "python", "freetoken")  # 2026-11-14: was a second ENGINE_PY, which hid the Python path above


def engine_archs() -> set[str] | None:
    """2026-11-14: the GGUF architectures this engine can load (GGUF_ARCH_TO_REGISTRY in models/gguf/config.py), read as text
    so ft-fit stays stdlib-only. None when the file or the dict is not found (then the check is skipped, not failed)."""
    try:
        src = open(os.path.join(ENGINE_SRC, "models", "gguf", "config.py"), encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r"GGUF_ARCH_TO_REGISTRY[^=]*=\s*\{(.*?)\n\}", src, re.S)
    return set(re.findall(r'^\s*"([^"]+)"\s*:', m.group(1), re.M)) or None if m else None


def engine_types() -> dict[int, str] | None:
    """2026-11-14: ggml tensor type ids the engine's GGUF reader knows (GGMLQuantizationType in third_party/gguf/constants.py).
    Knowing a type is not proof a kernel runs it, but an id missing here cannot load."""
    try:
        src = open(os.path.join(ENGINE_SRC, "third_party", "gguf", "constants.py"), encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r"class GGMLQuantizationType\(IntEnum\):\n(.*?)\n(?=\S)", src, re.S)
    return {int(v): k for k, v in re.findall(r"^\s+([A-Z0-9_]+)\s*=\s*(\d+)", m.group(1), re.M)} or None if m else None


def preflight(inv: dict) -> dict:
    """2026-11-14: checks before the first GPU load, from the model file and the engine's own tables, so a model fails in
    seconds with a reason instead of after a load (gemma4 IQ3_XS and LFM2 Q5_K_L failed on a fixed 5000 expert slots their
    shape could never hold). stop = a load cannot work; warn = it may, but a number below is off."""
    stop, warn = [], []
    archs = engine_archs()
    if archs is None:
        warn.append("could not read the engine's architecture list (models/gguf/config.py); arch check skipped")
    elif inv["arch"] not in archs:
        stop.append(f"architecture '{inv['arch']}' is not in this engine's GGUF list ({', '.join(sorted(archs))}). "
                    "A new architecture needs engine support first: check the engine and llama.cpp updates for it")
    known = engine_types()
    if known is None:
        warn.append("could not read the engine's tensor type table (third_party/gguf/constants.py); type check skipped")
    else:
        bad = [t for t in inv.get("tensor_types") or [] if t not in known]
        if bad:
            stop.append(f"tensor type id(s) {', '.join(map(str, bad))} are not in the engine's GGUF type table: a newer quant format "
                        "than this engine reads. Pick another quant, or update the engine")
        new = [known[t] for t in inv.get("tensor_types") or [] if t in known and type_name(t) == str(t)]
        if new:
            warn.append(f"tensor type(s) {', '.join(new)} are new to ft-fit's reader (scripts/gguf_header.py): the engine knows them, "
                        "but no fit here has run them yet")
    es = inv.get("expert_slots")
    if es and es < 5000:
        warn.append(f"{es} expert slots in all ({es // int(inv.get('experts') or 1)} expert layers x {inv.get('experts')}): under the probe's "
                    "5000, so runs use --moe-cache-auto and every slot count is capped at it")
    if inv.get("sliding_window") and inv.get("kv_bytes_per_token"):
        warn.append(f"sliding-window attention (window {inv['sliding_window']}): the KV estimate counts every layer at full "
                    "context, so it is high; the loads report the real KV")
    return {"stop": stop, "warn": warn}


def slot_fit(inv: dict, counts) -> list[int]:
    """2026-11-14: the slot counts at or under the model's expert total (a count above it cannot load)."""
    es = inv.get("expert_slots")
    return [n for n in counts if not es or n <= es]


def slots_room_in(text: str) -> int | None:
    """2026-11-14: "MoE cache of 5000 slots exceeds what the pools can hold (3840 slots)" -> 3840."""
    m = re.findall(r"exceeds what the pools can hold \((\d+) slots\)", text or "")
    return int(m[-1]) if m and int(m[-1]) > 0 else None


def slots_room(r: dict | None) -> int | None:
    """The slot room of a failed run: from the row, or (rows saved before 2026-11-14) from its error or log."""
    if not r or r.get("ok"):
        return None
    if r.get("slots_room"):
        return r["slots_room"]
    n = slots_room_in(str(r.get("error")))
    if n is None and r.get("name"):
        try:
            n = slots_room_in(open(os.path.join(PROBE_OUT, f"{r['name']}.log"), encoding="utf-8", errors="replace").read())
        except OSError:
            pass
    return n


def ladder(inv: dict, a) -> dict:
    R = Runner(inv["model"], a.max_runs, a.load_timeout, a.tag, a.ctx, a.fresh)
    if a.reserve_gib:
        R.base_env["FREETOKEN_DESKTOP_RESERVE_GB"] = str(a.reserve_gib)
    notes: list[str] = []
    best = None
    # stage 3 + 4: strategy, then memory ratio (each ratio run is also the strategy check)
    strategies = ["fused", "hybrid"] if inv["verdict"] == "fused" else ["hybrid"]

    def kv_short(r: dict) -> bool:
        return bool(r.get("kv_short"))  # the engine log says "Not enough memory for KV cache"

    def ram_short(r: dict) -> bool:
        return "available RAM" in str(r.get("error"))  # the guard stopped it on its host RAM floor

    if not inv.get("experts"):
        # A dense model: the engine ignores --moe-strategy and the expert cache, so fused/hybrid/ratio are one config.
        # What decides the KV room is memory-ratio x free VRAM minus the weights and the GDN state pool (4 slots per
        # running request plus a snapshot cache), so squeeze those, cheapest for quality first; the first that loads wins.
        # (--max-seq-len-override is only a cap on KV, so a smaller context does not help here.)
        steps = [("r90", ["--memory-ratio", "0.90"]),
                 ("r90-mr1", ["--memory-ratio", "0.90", "--max-running-requests", "1"]),
                 ("r90-mr1-naive", ["--memory-ratio", "0.90", "--max-running-requests", "1", "--cache-type", "naive"]),
                 ("r95-mr1-naive", ["--memory-ratio", "0.95", "--max-running-requests", "1", "--cache-type", "naive"]),
                 ("r95-mr1-naive-embhost", ["--memory-ratio", "0.95", "--max-running-requests", "1", "--cache-type", "naive", "--embed-weights", "host"])]
        R.max_runs += len(steps) - 1  # only the failed steps cost extra: the later stages (MTP, confirm, context) keep their budget
        for label, dargs in steps:
            r = R.run("4-dense", f"dense-{label}", dargs, base_args=False)
            if r is None:
                break
            if r["ok"]:
                best = r
                break
            notes.append(f"dense {' '.join(dargs)} failed: {err(r)}")
            if not kv_short(r):
                break  # not a KV shortfall: squeezing more will not help
        strategies = []
    for strat in strategies:
        base = ["--moe-strategy", strat]
        last = None
        # fused holds weights + KV inside the ratio, so start high; hybrid fills spare VRAM with experts, so start low
        for ratio in (sorted(RATIOS, reverse=True) if strat == "fused" else RATIOS):
            r = R.run("4-ratio", f"{strat}-r{int(ratio * 100)}", base + ["--memory-ratio", f"{ratio:.2f}"], base_args=strat != "fused")
            if r is None:
                break
            room = slots_room(r) if strat == "hybrid" else None
            if room:  # 2026-11-14: gemma4 IQ3_XS (3840) and LFM2 Q5_K_L (2432) never loaded on the probe's fixed 5000 slots
                notes.append(f"{strat} at ratio {ratio:.2f}: the 5000-slot expert cache does not fit (the engine has room for {room}); retried at {room}")
                R.max_runs += 1  # the retry does not eat the later stages' budget
                again = R.run("4-ratio", f"{strat}-r{int(ratio * 100)}-s{room}", base + ["--memory-ratio", f"{ratio:.2f}", "--moe-cache-size", str(room)])
                if again is None:
                    break
                r = again
            if not r["ok"]:
                notes.append(f"{strat} at ratio {ratio:.2f} failed: {err(r)}")
                if kv_short(r) and strat == "hybrid" and last is None:
                    continue  # hybrid starts at the lowest ratio: a higher one leaves more room for KV
                if ram_short(r) and strat == "hybrid" and last is None and not r.get("gpu_reset"):
                    # 2026-11-15: a higher ratio keeps more experts in VRAM, so fewer in host RAM (qwen38 IQ3_M: min RAM 3.23 GiB
                    # at 0.80, 3.62 at 0.85). occamy IQ3_M stopped here at 0.80 with 0.52 GiB, 0.18 under the floor, and never
                    # tried 0.85. Wait for the guard-freed RAM to come back first. Undo: delete this branch.
                    print(f"      .. host RAM ran out at ratio {ratio:.2f}: waiting {RAM_SETTLE_S} s, then a higher ratio", flush=True)
                    time.sleep(RAM_SETTLE_S)
                    continue
                break
            if best is None or better(r, best):
                best = r
            if strat == "fused":
                break  # the highest ratio that loads is the answer; lower only shrinks KV
            if last is not None and not better(r, last):
                break  # a higher ratio no longer helps
            last = r
        if best is not None and strat == "fused":
            break  # fused works; hybrid only matters when fused does not
    if best is None and "hybrid" in strategies:
        # 2026-11-14: MoE squeeze ladder, like the dense one: when no ratio loads, give KV and the state pool more room, cheapest
        # first; the first that loads wins. (No vision or context step: ladder runs have no vision, and the context is only a KV cap.)
        fails = [r for r in R.log if r["stage"] == "4-ratio" and not r.get("ok")]
        ram = any("available RAM" in str(r.get("error")) for r in fails)
        ratio = f"{max(RATIOS) if fails and all(kv_short(r) for r in fails) else min(RATIOS):.2f}"
        base = ["--moe-strategy", "hybrid", "--memory-ratio", ratio]
        rooms = [n for n in (slots_room(r) for r in fails) if n]
        cap = min(rooms) if rooms else None  # 2026-11-14: the smallest slot room a ratio run reported
        sq = min([3000] + ([cap] if cap else []) + ([inv["expert_slots"]] if inv.get("expert_slots") else []))  # 2026-11-14: and the expert total
        steps = [("mr1" + (f"-s{cap}" if cap else ""), ["--max-running-requests", "1"] + (["--moe-cache-size", str(cap)] if cap else [])),
                 (f"mr1-s{sq}", ["--max-running-requests", "1", "--moe-cache-size", str(sq)]),
                 (f"mr1-s{sq}-naive", ["--max-running-requests", "1", "--moe-cache-size", str(sq), "--cache-type", "naive"])]
        if cap == sq:
            steps.pop(1)  # same as the first step
        if ram:
            notes.append("MoE squeeze skipped: host RAM ran out (the guard stopped a run), and the squeeze steps only free VRAM")
        else:
            R.max_runs += len(steps)
            for label, sargs in steps:
                r = R.run("4-squeeze", f"hybrid-r{int(float(ratio) * 100)}-{label}", base + sargs)
                if r is None:
                    break
                if r["ok"]:
                    best = r
                    notes.append(f"MoE squeeze: loaded with `{' '.join(sargs)}` at ratio {ratio}")
                    break
                notes.append(f"MoE squeeze {' '.join(sargs)} failed: {err(r)}")
                if "available RAM" in str(r.get("error")):
                    break
    if best is None:
        notes.append("no config loaded; see the errors above")
        return {"inv": inv, "ctx_used": R.ctx, "best": None, "spec": None, "flags": [], "confirm": None, "notes": notes, "log": R.log, "runs": R.runs, "reused": R.reused}
    plain = best
    # 2026-11-14: the context is a goal (--ctx-goal, default --ctx or 65536) with a hard floor under it (--ctx-floor, default
    # 32768): a fit between the two is "under goal", not failed. A floor over the goal raises the goal to it.
    goal = int(getattr(a, "ctx_goal", 0) or a.ctx or 65536)
    floor = int(getattr(a, "ctx_floor", 0) or min(32768, goal))
    goal = max(goal, floor)
    if getattr(a, "tune", False):
        # --tune (2026-11-13): the machine-level settings, once per model class. Flags (each winner run twice) and the parallel
        # report on the plain ratio winner; saved to speedprobe/tune/<class>.json, which every later fit of the class reads.
        flags_out = flag_stage(R, inv, plain)
        par = None if getattr(a, "no_parallel", False) else parallel_stage(R, plain, notes)
        notes.append(f"tune run: flags and parallel report saved to {save_tune(inv, plain, flags_out, par)}; not a config to stamp")
        return {"inv": inv, "ctx_used": R.ctx, "best": plain, "spec": None, "use": plain, "flags": flags_out, "confirm": None, "parallel": par,
                "tune": True, "reserve": a.reserve_gib or None, "notes": notes, "log": R.log, "runs": R.runs, "reused": R.reused}
    # stage 5: spec decoding. MTP first; 2026-11-12: then a DFlash draft when --dflash-head names one. The faster is `spec`.
    spec = None

    def dflash_ks(head: str) -> tuple[int, ...]:
        # 2026-11-13: DFlash drafts a whole block; k 2-4 left most of it unused (11/7 run: DFlash k3 107 vs MTP k3 128 tok/s).
        # Try up to block_size - 1 (the engine's cap). Undo: ft-fit.py.bak-20261113-dflashk
        try:
            with open(os.path.join(head, "config.json"), encoding="utf-8") as f:
                c = json.load(f)
            b = int((c.get("dflash_config") or {}).get("block_size") or c.get("block_size") or 8)
        except (OSError, ValueError):
            b = 8
        return tuple(k for k in (3, 5, 7, 11, 15) if k <= b - 1)

    def spec_try(kind: str, extra: list[str], ks: tuple[int, ...] = (2, 3, 4)) -> dict | None:
        top = None
        for k in ks:
            r = R.run("5-spec", f"{plain['label']}-{kind}{k}", plain["args"] + ["--speculative-algorithm", kind, "--speculative-num-steps", str(k)] + extra)
            if r is None or not r["ok"]:
                if r is not None:
                    notes.append(f"{'MTP' if kind == 'mtp' else 'DFlash'} k={k} failed: {err(r)}")
                break
            if top is None or better(r, top):
                top = r
            elif tps(r) < tps(top) and kind == "mtp":  # DFlash speed is not monotone in k: run every rung
                break
        return top

    draft = [] if inv["builtin_mtp"] else (["--speculative-draft-model-path", inv["sidecars"][0]] if inv["sidecars"] else None)
    if draft is None:
        notes.append("no MTP draft head on disk; " + ("MTP skipped" if getattr(a, "dflash_head", None) else "spec decoding skipped"))
    else:
        spec = spec_try("mtp", draft)
        tm = token_map(inv["model"], getattr(a, "token_map", "off"), notes) if spec is not None else None
        if tm:  # 2026-11-13 (upstream #892): the draft scores only the ids in the map; the target still verifies the whole vocabulary
            R.max_runs += 1
            r = R.run("5-spec", f"{spec['label']}-tmap", spec["args"] + ["--speculative-token-map", tm])
            if r is not None:
                gain = r["ok"] and decide(R, r, spec, notes, "token map")
                notes.append(f"token map ({os.path.basename(tm)}): " + (f"{tps(r)} vs {tps(spec)} tok/s, {'kept' if gain else 'not kept'}" if r["ok"] else f"failed: {err(r)}"))
                if gain:
                    spec = r
    dhead = getattr(a, "dflash_head", None)
    if dhead and not os.path.exists(dhead):
        notes.append(f"--dflash-head {dhead} does not exist; DFlash skipped")
    elif dhead:
        dks = dflash_ks(dhead)
        R.max_runs += len(dks)  # its own runs: the later stages keep their budget
        notes.append(f"DFlash ladder k={list(dks)} (draft block size from its config.json)")
        dfl = spec_try("dflash", ["--speculative-draft-model-path", dhead], dks)
        if dfl is not None and spec is not None:
            dwin = decide(R, dfl, spec, notes, "DFlash vs MTP")
            notes.append(f"spec decoding: MTP {tps(spec)} tok/s, DFlash {tps(dfl)} tok/s; {'DFlash' if dwin else 'MTP'} kept")
            if dwin:
                spec = dfl
        elif dfl is not None:
            spec = dfl
    # 2026-11-13: `use` is the config the card will carry. Slots are traded on it (after spec, which takes VRAM), not on plain.
    use = spec if spec and decide(R, spec, plain, notes, "spec vs plain decode") else plain
    slots = None
    if "--moe-cache-size" in use["args"]:  # 2026-11-14: the squeeze ladder already cut the slots to load; the sweep's 5000 ref failed
        notes.append("slot sweep skipped: the squeeze ladder or the slot-room retry fixed the slot count (" + use["args"][use["args"].index("--moe-cache-size") + 1] + ")")
    elif inv.get("experts") and "hybrid" in use["args"] and not a.no_slots:
        # 2026-11-13: a model with at most 5000 experts can't run the sweep (5000 and up exceed its pools), so it gets auto too
        few = int(inv.get("expert_slots") or int(inv.get("layers") or 0) * int(inv.get("experts") or 0)) <= 5000
        if getattr(a, "slots_mode", "sweep") == "auto" or few:  # 2026-11-12: one --moe-cache-auto run instead of the sweep
            slots = slot_auto_stage(R, inv, use, notes, int(getattr(a, "kv_reserve", 0) or goal))
        else:
            slots = slot_stage(R, inv, use, notes, goal)
        if slots.get("row"):
            use = slots.pop("row")
    # flags and the parallel report come from the class's tune file (2026-11-13); a fit no longer runs them itself
    tune = load_tune(inv)
    flags_out = [dict(f) for f in (tune or {}).get("flags", []) if f.get("keep")]
    par = (tune or {}).get("parallel")
    if tune:
        notes.append(f"flags from the tune run of {tune.get('when')} on {os.path.basename(str(tune.get('model')))} (class {tune.get('class')}): "
                     + (", ".join(f"`{f['value']}`" if f["kind"] == "arg" else f"{f['name']}={f['value']}" for f in flags_out) or "none kept"))
    else:
        notes.append(f"no tune result for class {model_class(inv)}: no tune flags applied. Run once with --tune (BENCH_FT_FIT_TUNE=1).")
    for d in CLASS_DEFAULTS.get("moe" if inv.get("experts") else "dense", []):  # 2026-11-14: unless the tune kept a value of the same flag
        if not any(f["kind"] == d["kind"] and f["name"] == d["name"] for f in flags_out):
            flags_out.append(dict(d))
            notes.append(f"class default applied: `{d['value']}` ({d['result']})")
    # stage 7: the config as stamped, in one run: winner + kept flags, vision if present, a 5K prompt, at the model's own
    # maximum context (so its KV is the real room). Its numbers are the card's.
    vis = bool(inv["mmproj"])
    cap = min(inv.get("context_length") or 262144, 524288) // 1024 * 1024
    R.max_runs += 3

    def final(args: list[str], env: dict, tag: str, vision: bool) -> tuple[dict | None, list[str], int]:
        r, c = None, cap
        for attempt in range(3):
            r = R.run("7-final", f"{use['label']}-final{tag}{attempt}{'-vis' if vision else ''}", args, tests="decode,p5k", vision=vision, env=env or None,
                      base_args="fused" not in args, ctx=c)
            if r is None or r["ok"]:
                break
            notes.append(f"as-stamped run {'with vision ' if vision else ''}at {' '.join(args)} (context {c}) failed: {err(r)}")
            m = [i for i, x in enumerate(args) if x == "--memory-ratio"]
            if r.get("kv_short"):
                c = c * 3 // 4 // 1024 * 1024  # the KV does not fit at this cap: a smaller one
            elif m:
                args = list(args)
                args[m[0] + 1] = f"{float(args[m[0] + 1]) - 0.05:.2f}"
            else:
                break
        return r, args, c

    keep_env = {f["name"]: f["value"] for f in flags_out if f["kind"] == "env"}
    keep_args = [x for f in flags_out if f["kind"] == "arg" for x in f["value"].split()]
    confirm, fargs, fcap = final(use["args"] + keep_args, keep_env, "", vis)
    if vis and confirm is not None and not confirm["ok"]:
        confirm, fargs, fcap = final(use["args"] + keep_args, keep_env, "-novis", False)
        notes.append("vision does not fit with this config: launch with --mm-disable vision")
    if flags_out and confirm is not None and (not confirm["ok"] or tps(confirm) < tps(use) * (1 - FLAG_WORSE)):
        # the tune flags hurt this model (or broke the load): one run without them, and the better of the two is stamped
        R.max_runs += 1
        bare, bargs, bcap = final(list(use["args"]), {}, "-noflags", bool(confirm.get("vision")) if confirm["ok"] else vis)
        if bare is not None and bare["ok"] and (not confirm["ok"] or better(bare, confirm)):
            notes.append(f"tune flags dropped: {tps(bare)} tok/s without them, {tps(confirm) if confirm['ok'] else 'a failed load'} with them")
            confirm, fargs, fcap, flags_out, keep_env = bare, bargs, bcap, [], {}
    if fargs[:len(use["args"])] != use["args"]:  # the as-stamped run had to lower the memory ratio
        use = dict(use, args=fargs[:len(use["args"])])
    vis_ok = vis and bool(confirm and confirm.get("ok") and confirm.get("vision"))
    stamped = dict(use, args=fargs)
    # stage 8: context on the as-stamped config (its load is the run above), then the 8-bit KV row
    ctxmax = ctx_stage(R, inv, stamped, vis_ok, notes, first=confirm if confirm and confirm.get("ok") else None, env=keep_env or None, cap=fcap)
    if ctxmax.get("ctx") and ctxmax["ctx"] < min(floor, fcap):
        notes.append(f"context {ctxmax['ctx']} is under the floor of {floor} tokens: this config fails the context check")
    elif ctxmax.get("ctx") and ctxmax["ctx"] < min(goal, fcap):
        notes.append(f"context {ctxmax['ctx']} is under the goal of {goal} tokens but over the floor of {floor}: usable, labeled under goal")
    kv8 = None
    # 2026-11-14: 8-bit KV is a rescue, not a report on every fit: it runs only when the 16-bit KV misses the floor (or with
    # FT_FIT_KV8=1 / FT_FIT_KV8_SLOTS). It is lossy (ctx_rot on the 35B distill: 8K 7/10 vs 9/10 at 16-bit, and a hang in a
    # 30K prefill), so it never goes in the launch line: the card shows it as a separate, labeled line when it holds more.
    missed = (ctxmax.get("ctx") or 0) < min(floor, fcap)
    if confirm and confirm.get("ok") and not getattr(a, "no_kv8", False) and not (missed or os.environ.get("FT_FIT_KV8") == "1" or os.environ.get("FT_FIT_KV8_SLOTS")):
        notes.append("8-bit KV not run: the 16-bit KV holds the context floor" + (
            f" ({floor}), not the goal ({goal}); FT_FIT_KV8=1 runs it (lossy)" if (ctxmax.get("ctx") or 0) < min(goal, fcap) else ""))
    elif confirm and confirm.get("ok") and not getattr(a, "no_kv8", False):
        kv8 = kv8_row(R, stamped, vis_ok, keep_env or None, fcap, confirm, ctxmax)
        row8 = kv8.pop("_row", None)
        kv8["rescue"] = missed
        base_kv = kv8.get("base_kv_tokens") or 0
        if missed and kv8.get("ok") and (kv8.get("kv_tokens") or 0) > base_kv * 1.2:
            # it really holds more: confirm with a prompt that fills it (fp8 has hung on a long prefill), on its own labels
            kwin = dict(stamped, label=stamped["label"] + "-kv8", args=stamped["args"] + ["--kv-cache-dtype", KV8])
            kc = ctx_stage(R, inv, kwin, vis_ok, notes, first=row8, env=keep_env or None, cap=fcap)
            kv8.update(ctx=kc.get("ctx"), ctx_ok=kc.get("ok"), floor_ok=bool(kc.get("ok")) and (kc.get("ctx") or 0) >= min(floor, fcap),
                       goal_ok=bool(kc.get("ok")) and (kc.get("ctx") or 0) >= min(goal, fcap), launch=kwin["args"])
            notes.append(f"8-bit KV rescue: {kc.get('ctx')} tokens " + ("confirmed by a fill test" if kc.get("ok") else "NOT confirmed")
                         + f" (16-bit: {ctxmax.get('ctx')}); floor {floor} " + ("met" if kv8["floor_ok"] else "still missed") + "; lossy, see the card")
        elif missed and kv8.get("ok"):
            notes.append(f"8-bit KV rescue: no real gain ({kv8.get('kv_tokens')} KV tokens vs {base_kv} at 16-bit); not offered")
        # 2026-11-14: FT_FIT_KV8_SLOTS=5500,6000 reruns those slot counts with 8-bit KV (half the KV per token, so more experts
        # fit over the floor). A report only, like the kv8 row: never the winner.
        ks = slot_fit(inv, [int(x) for x in os.environ.get("FT_FIT_KV8_SLOTS", "").split(",") if x.strip()])
        if ks and kv8.get("ok"):
            kcap = min(inv.get("context_length") or 262144, 524288) // 1024 * 1024
            R.max_runs += len(ks)
            kv8["slots"] = []
            seen: dict[int, int] = {}
            for n in ks:  # a count listed again is a repeat run (label -r2, -r3, ...), for the spread
                seen[n] = seen.get(n, 0) + 1
                rep = f"-r{seen[n]}" if seen[n] > 1 else ""
                r = R.run("8-kv8s", f"{stamped['label']}-kv8-s{n}{rep}",stamped["args"] + ["--kv-cache-dtype", KV8, "--moe-cache-size", str(n)],
                          tests="decode,p5k", env=keep_env or None, ctx=kcap)
                if r is None:
                    break
                kv8["slots"].append({"slots": n, "ok": r["ok"], "decode": tps(r) if r["ok"] else None, "prefill": (r.get("prefill") or {}).get("p5k"),
                                     "kv_tokens": r.get("kv_tokens"), "floor_ok": (r.get("kv_tokens") or 0) >= floor, "goal_ok": (r.get("kv_tokens") or 0) >= goal,
                                     "error": None if r["ok"] else str(r.get("error"))[:200]})
    return {"inv": inv, "ctx_used": R.ctx, "best": plain, "spec": spec, "use": use, "flags": flags_out, "confirm": confirm, "ctxmax": ctxmax,
            "slots": slots, "parallel": par, "floor": floor, "goal": goal, "kv8": kv8, "tune_from": {k: tune.get(k) for k in ("when", "model", "class")} if tune else None,
            "reserve": a.reserve_gib or None, "notes": notes, "log": R.log, "runs": R.runs, "reused": R.reused}


def model_class(inv: dict) -> str:
    """The class a tune result is shared by: architecture, and expert model or dense."""
    return re.sub(r"[^a-z0-9]+", "-", f"{inv.get('arch')}-{'moe' if inv.get('experts') else 'dense'}".lower())


def save_tune(inv: dict, plain: dict, flags: list[dict], par: dict | None) -> str:
    os.makedirs(TUNE_DIR, exist_ok=True)
    p = os.path.join(TUNE_DIR, model_class(inv) + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"class": model_class(inv), "model": inv["model"], "when": time.strftime("%Y-%m-%d %H:%M"), "plain_args": plain["args"],
                   "plain_tps": tps(plain), "flags": flags, "parallel": par}, f, indent=1)
    return p


def load_tune(inv: dict) -> dict | None:
    try:
        with open(os.path.join(TUNE_DIR, model_class(inv) + ".json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def token_map(model: str, how: str, notes: list[str]) -> str | None:
    """A --speculative-token-map file for this model: `how` is off, a file, or auto (speedprobe/token-maps/<model>.bin, written
    once by upstream's scripts/make_token_map.py with the engine's Python; it reads only the tokenizer, no GPU)."""
    if not how or how == "off":
        return None
    if how != "auto":
        if os.path.isfile(how):
            return how
        notes.append(f"token map {how} does not exist; skipped")
        return None
    p = os.path.join(MAP_DIR, slug(model) + ".bin")
    if os.path.isfile(p):
        return p
    if not os.path.isfile(ENGINE_PY):
        notes.append(f"token map skipped: the engine's Python is not at {ENGINE_PY} (set FT_FIT_PYTHON)")
        return None
    os.makedirs(MAP_DIR, exist_ok=True)
    try:
        q = subprocess.run([ENGINE_PY, "scripts/make_token_map.py", model, "-o", p], cwd=REPO, env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "python")),
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        tail = q.stdout.decode("utf-8", "replace").strip().splitlines()[-1:]
    except (OSError, subprocess.TimeoutExpired) as e:
        q, tail = None, [str(e)]
    if q is None or q.returncode or not os.path.isfile(p):
        notes.append(f"token map could not be written: {(tail or ['no output'])[0][:200]}")
        return None
    return p


def kv8_row(R: "Runner", win: dict, vis: bool, env: dict | None, cap: int, confirm: dict, ctxmax: dict) -> dict:
    """2026-11-13 (upstream #894): the as-stamped config once more with the KV stored in 8 bits. A report only: it is lossy, so
    it never goes in the launch line; the card shows how much more context it would hold and what it does to speed and text."""
    R.max_runs += 1
    out = {"format": KV8, "ok": False, "base_kv_tokens": ctxmax.get("kv_tokens") or confirm.get("kv_tokens")}
    r = R.run("8-kv8", f"{win['label']}-kv8", win["args"] + ["--kv-cache-dtype", KV8], tests="decode", vision=vis, env=env,
              base_args="fused" not in win["args"], ctx=cap)
    if r is not None:
        out.update(ok=bool(r["ok"]), kv_tokens=r.get("kv_tokens"), decode=tps(r) if r["ok"] else None, base_decode=tps(confirm),
                   text_same=bool(r["ok"] and r.get("text") == confirm.get("text")), max_vram_gib=r.get("max_vram_gib"),
                   error=None if r["ok"] else str(r.get("error"))[:300], _row=r)
    return out


def flag_tests(fl: dict) -> str:
    """A flag entry's probe tests: its "tests" (default "decode"), plus decode (the text check) and its metric's test."""
    ts = [t for t in str(fl.get("tests") or "decode").split(",") if t]
    mt = str(fl.get("metric") or "decode").partition(":")[2]
    return ",".join((["decode"] if "decode" not in ts else []) + ts + ([mt] if mt and mt not in ts else []))


def flag_metric(r: dict | None, metric: str) -> float:
    """decode: decode tok/s of the decode test; prefill:<test>: that test's prefill tok/s; total:<test>: its decode_tps (cN: all requests)."""
    if not r or not r.get("ok"):
        return 0.0
    kind, _, t = metric.partition(":")
    v = tps(r) if kind == "decode" else ((r.get("prefill") if kind == "prefill" else r.get("tps")) or {}).get(t)
    return float(v or 0)


def flag_stage(R: "Runner", inv: dict, plain: dict) -> list[dict]:
    """Stage 6, generalized 2026-11-12. Each safe entry runs once per value ("values", or the single "value") with its "tests",
    and is scored by its "metric" against the plain config run with the same tests (the plain row itself for "decode", else one
    reference run per tests string). Kept: lossless true, decode text identical to plain's, metric over SAME better, and neither
    decode tok/s nor the 5K prefill more than FLAG_WORSE under the reference. Of an entry's values, and of kept entries with
    the same name, only the best is kept. The kept value is written back as the entry's "value", which card() and winner() read."""
    flags, out, refs = load_flags(), [], {"decode": plain}
    values = lambda fl: [str(v) for v in fl["values"]] if fl.get("values") else [str(fl.get("value", ""))]
    todo = [fl for fl in flags if fl.get("status") == "safe" and (inv.get("experts") or "moe" not in fl["name"].lower())]
    # what this stage costs beyond one run per entry: the extra values and the reference runs
    R.max_runs += sum(len(values(fl)) - 1 for fl in todo) + len({flag_tests(fl) for fl in todo} - {"decode"})
    spent = False
    for fl in flags:
        if not inv.get("experts") and "moe" in fl["name"].lower():
            out.append({**fl, "result": "not run (MoE-only flag; dense model)"})
            continue
        if fl.get("status") == "handled":
            continue  # 2026-11-14: run by its own stage (kv8 rescue, token map), so the card's flag list leaves it out
        if fl.get("status") != "safe":
            out.append({**fl, "result": f"not run ({fl.get('status')})"})
            continue
        tests, metric = flag_tests(fl), str(fl.get("metric") or "decode")
        if tests not in refs:
            refs[tests] = R.run("6-flag", f"{plain['label']}-ref-{tests.replace(',', '-')}", plain["args"], tests=tests)
        ref = refs[tests]
        if ref is None:
            break
        if not ref.get("ok"):
            out.append({**fl, "result": f"not run (the plain reference with tests {tests} failed: {err(ref)[:200]})"})
            continue
        ref_m, fid, rows, said = flag_metric(ref, metric), re.sub(r"[^a-z0-9]+", "", fl["id"].lower()), [], []

        def one(label: str, v: str) -> dict | None:
            r = R.run("6-flag", label, plain["args"] + ([] if fl["kind"] == "env" else v.split()), tests=tests,
                      env={fl["name"]: v} if fl["kind"] == "env" else None)
            if r is None:
                return None
            ok = bool(r.get("ok"))
            same, m = ok and r.get("text") == plain.get("text"), flag_metric(r, metric)
            checks = [("decode", tps(r), tps(ref)), ("5K prefill", (r.get("prefill") or {}).get("p5k"), (ref.get("prefill") or {}).get("p5k"))]
            worse = [f"{n} {new} vs {old}" for n, new, old in checks if ok and new and old and new < old * (1 - FLAG_WORSE)]
            said.append((f"`{v}`: " if fl.get("values") else "") + fmt(r) + ("" if metric == "decode" or not ok else f"; {metric} {m} vs {ref_m}")
                        + ("; text identical" if same else ("; TEXT DIFFERS" if ok else "")) + (f"; worse: {', '.join(worse)}" if worse else ""))
            return {"label": label, "value": v, "ok": ok, "metric": m if ok else None, "ref": ref_m, "text_same": same, "worse": worse,
                    "pass": bool(fl.get("lossless") is True and same and ref_m and m > ref_m * (1 + SAME) and not worse)}

        for v in values(fl):
            row = one(f"{plain['label']}-{fid}" + ("-" + re.sub(r"[^a-z0-9.]+", "", (v.split() or ["x"])[-1].lower()) if fl.get("values") else ""), v)
            if row is None:
                spent = True
                break
            rows.append(row)
        if rows:
            top = max(rows, key=lambda x: (x["pass"], x["ok"], x["metric"] or 0))
            if top["pass"]:  # 2026-11-13: one 512-token run is +-2%, so a winner has to win twice
                R.max_runs += 1
                said.append("second run")
                again = one(top["label"] + "-again", top["value"])
                if again is None or not again["pass"]:
                    top["pass"] = False
                    said.append("gain not repeated" if again else "second run skipped (run budget): not kept")
                else:
                    top["metric"] = round((top["metric"] + again["metric"]) / 2, 2)
            why = "" if fl.get("lossless") is True else f"; lossless {fl.get('lossless')}: reported only"
            out.append({**fl, "value": top["value"], "result": " | ".join(said) + why, "keep": top["pass"], "metric_value": top["metric"],
                        "ref_value": ref_m, "variants": rows})
        if spent:
            break
    kept: dict = {}
    for f in out:  # two kept entries of one flag (--moe-cpu-threads 6 and 8): only the bigger gain stays
        if f.get("keep"):
            o = kept.get((f["kind"], f["name"]))
            lose = f if o and o["metric_value"] / o["ref_value"] >= f["metric_value"] / f["ref_value"] else o
            if lose:
                lose["keep"] = False
                lose["result"] += f"; not kept: `{(f if lose is o else o)['id']}` gains more"
            if lose is not f:
                kept[(f["kind"], f["name"])] = f
    return out


def parallel_stage(R: "Runner", plain: dict, notes: list[str]) -> dict:
    """Stage 6p (2026-11-12): N requests at once on the plain config, N in PARALLEL, stopping at the first N that fails. The probe's
    cN test gives the total tok/s over the N requests and each request's own. best_n is the N with the highest total when
    that beats one request's decode by more than SAME, else 1. It only reports: how much context each request gets is the user's call."""
    # 2026-11-14: rows are (requests, graph bs, extra args). The graph series stops at its first failure. Then one row at 4 requests
    # with graph bs 1, which is what the live launchers run today (FT_GRAPH_MAX_BS=1), and with FT_FIT_PAR_EXTRA (e.g. the spec
    # args) the 2 and 4 rows again with those args added. Only the plain graph series sets best_n.
    extra = os.environ.get("FT_FIT_PAR_EXTRA", "").split()
    series = [(n, n, []) for n in PARALLEL]
    tail = [(4, 1, [])] + ([(2, 2, extra), (4, 4, extra)] if extra else [])
    out = {"rows": [], "best_n": 1, "plain_tps": tps(plain), "extra": " ".join(extra) or None}
    R.max_runs += len(series) + len(tail)
    base = list(plain["args"])
    while "--max-running-requests" in base:  # a dense winner may carry --max-running-requests 1
        i = base.index("--max-running-requests")
        del base[i:i + 2]
    failed = False
    for n, g, x in series + tail:
        if failed and g == n and not x:
            continue  # the graph series already failed at a smaller N
        t = f"c{n}"
        label = f"{plain['label']}-par{n}" + ("" if g == n else f"-g{g}") + ("-x" if x else "")
        r = R.run("6p-parallel", label, base + x + ["--max-running-requests", str(n), "--cuda-graph-max-bs", str(g)],
                  tests=f"decode,{t}", env={"FT_GRAPH_MAX_BS": str(g)})
        if r is None:
            notes.append(f"parallel stage: {label} skipped, run budget spent")
            break
        ok = bool(r.get("ok"))
        out["rows"].append({"n": n, "graph_bs": g, "extra": " ".join(x) or None, "ok": ok, "total_tps": (r.get("tps") or {}).get(t) if ok else None,
                            "per_req_tps": (r.get("per_req") or {}).get(t), "single_tps": tps(r) if ok else None, "kv_tokens": r.get("kv_tokens"),
                            "max_vram_gib": r.get("max_vram_gib"), "min_ram_gib": r.get("min_ram_gib"), "error": None if ok else str(r.get("error"))[:200]})
        if not ok:
            notes.append(f"parallel stage: {label} failed: {err(r)}")
            failed = failed or (g == n and not x)
    good = [x for x in out["rows"] if x["ok"] and x["total_tps"] and x["graph_bs"] == x["n"] and not x["extra"]]
    if good:
        top = max(good, key=lambda x: x["total_tps"])
        if top["total_tps"] > tps(plain) * (1 + SAME):
            out["best_n"] = top["n"]
    return out


def slot_auto_stage(R: "Runner", inv: dict, win: dict, notes: list[str], reserve: int) -> dict:
    """Stage 4s with --slots-mode auto (2026-11-12): one run where the engine sizes the expert slots itself (--moe-cache-auto) after
    holding `reserve` KV tokens back. The probe's base args are left out (their --moe-cache-size 5000 cannot go with
    --moe-cache-auto), so --moe-cpu-layers auto is passed here. When the run is good and its log names the slot count
    ("--moe-cache-auto resolved moe_cache_size=N"), the row becomes the plain winner as win args + --moe-cache-size N, which the
    later stages and the card use; otherwise the 5000-slot winner stays."""
    cap = min(inv.get("context_length") or 262144, 524288) // 1024 * 1024
    out = {"mode": "auto", "ref": 5000, "pick": 5000, "rows": [], "cap": cap, "kv_reserve": reserve}
    R.max_runs += 1
    r = R.run("6-slots", f"{win['label']}-sauto", ["--moe-cpu-layers", "auto"] + win["args"] + ["--moe-cache-auto", "--kv-reserve-tokens", str(reserve)],
              tests="decode,p5k", base_args=False, ctx=cap)
    if r is None:
        notes.append("auto slots skipped: run budget spent")
        return out
    n = r.get("slots_auto")
    out["rows"].append({"slots": n, "ok": r["ok"], "decode": tps(r), "prefill": (r.get("prefill") or {}).get("p5k"), "kv_tokens": r.get("kv_tokens"),
                        "max_vram_gib": r.get("max_vram_gib"), "error": None if r["ok"] else str(r.get("error"))[:200]})
    if not r["ok"]:
        notes.append(f"auto slots: the --moe-cache-auto run (KV floor {reserve} tokens) failed: {err(r)}; kept 5000 slots")
    elif not n:
        notes.append("auto slots: the run was fine but its log does not name the slot count the engine chose; kept 5000 slots")
    else:
        # 2026-11-13: the card keeps --moe-cache-auto + the reserve, not the slot count it resolved to: a fixed 2036 slots
        # left LFM2 62603 KV tokens on the next load (free VRAM moves), under the 65536 floor the auto run held.
        out.update(pick=n, kv_tokens=r.get("kv_tokens"), row=dict(r, args=win["args"] + ["--moe-cache-auto", "--kv-reserve-tokens", str(reserve)]))
    return out


def slot_stage(R: "Runner", inv: dict, win: dict, notes: list[str], floor: int = 0) -> dict:
    """Stage 6 (was 4s; 2026-11-13: after spec, against a context floor): the expert GPU slots against the KV on the config the
    card will carry. Runs load at the model's own maximum context, so "Allocating N tokens for KV cache" is the real room.
    The goal is the fastest config that holds `floor` tokens (the context goal, 2026-11-14):
      5000 slots hold the floor -> try SLOTS_UP while the floor still holds and decode gains over SAME;
      they do not               -> step down SLOTS until the KV reaches the floor (speed is not weighed: the floor is the goal);
                                   if no count reaches it, the count with the most KV is kept and a note says so.
    floor 0 is the 2026-11-11 rule: the lowest count within SLOT_DECODE / SLOT_PREFILL of 5000 while the KV grows.
    Returns the table, the pick, and (when the pick is not 5000) its row, which becomes the winner."""
    cap = min(inv.get("context_length") or 262144, 524288) // 1024 * 1024
    floor = min(floor, cap)
    out = {"ref": 5000, "pick": 5000, "rows": [], "cap": cap, "floor": floor or None}
    R.max_runs += len(SLOTS) + 1

    def one(n: int) -> dict | None:
        r = R.run("6-slots", f"{win['label']}-s{n}", win["args"] + (["--moe-cache-size", str(n)] if n != 5000 else []), tests="decode,p5k", ctx=cap)
        if r is not None:
            out["rows"].append({"slots": n, "ok": r["ok"], "decode": tps(r), "prefill": (r.get("prefill") or {}).get("p5k"),
                                "kv_tokens": r.get("kv_tokens"), "max_vram_gib": r.get("max_vram_gib"), "error": None if r["ok"] else str(r.get("error"))[:200]})
        return r

    ref = one(5000)
    if ref is None or not ref["ok"]:
        notes.append("slot sweep skipped: " + ("run budget spent" if ref is None else f"the 5000-slot run at context {cap} failed: {err(ref)}"))
        return out
    pick, kv = ref, ref.get("kv_tokens") or 0
    if floor:
        if kv >= floor:
            for n in slot_fit(inv, SLOTS_UP):
                r = one(n)
                if r is None:
                    break
                if not r["ok"]:
                    notes.append(f"slot sweep: {n} slots failed: {err(r)}; kept {out['pick']}")
                    break
                if (r.get("kv_tokens") or 0) < floor:
                    notes.append(f"slot sweep: {n} slots leave {r.get('kv_tokens')} KV tokens, under the goal of {floor}; kept {out['pick']}")
                    break
                if not decide(R, r, pick, notes, f"{n} slots vs {out['pick']}"):
                    notes.append(f"slot sweep: {n} slots is no faster ({tps(r)} vs {tps(pick)} tok/s); kept {out['pick']}")
                    break
                pick, kv, out["pick"] = r, r["kv_tokens"], n
        else:
            for n in slot_fit(inv, SLOTS):
                r = one(n)
                if r is None:
                    break
                if not r["ok"]:
                    notes.append(f"slot sweep: {n} slots failed: {err(r)}")
                    break
                if (r.get("kv_tokens") or 0) <= kv:
                    notes.append(f"slot sweep: {n} slots gave no more context ({r.get('kv_tokens')} KV tokens vs {kv}); kept {out['pick']}")
                    break
                pick, kv, out["pick"] = r, r["kv_tokens"], n
                if kv >= floor:
                    break
            if kv < floor:
                notes.append(f"slot sweep: no slot count holds the goal of {floor} tokens; kept {out['pick']} slots ({kv} KV tokens), the most context found")
            elif pick is not ref:
                notes.append(f"slot sweep: {out['pick']} slots hold the goal ({kv} KV tokens) at {tps(pick)} tok/s; 5000 slots held {ref.get('kv_tokens')} at {tps(ref)}")
        out["kv_tokens"] = kv
        if pick is not ref:
            out["row"] = pick
        return out
    if kv >= cap:
        notes.append(f"slot sweep: 5000 slots already hold the model's whole context ({cap}); nothing to trade")
        return out
    pre0 = (ref.get("prefill") or {}).get("p5k") or 0
    for n in slot_fit(inv, SLOTS):
        r = one(n)
        if r is None:
            break
        if not r["ok"]:
            notes.append(f"slot sweep: {n} slots failed: {err(r)}")
            break
        pre = (r.get("prefill") or {}).get("p5k") or 0
        if tps(r) < tps(ref) * (1 - SLOT_DECODE) or pre < pre0 * (1 - SLOT_PREFILL):
            notes.append(f"slot sweep: {n} slots is too slow (decode {tps(r)} vs {tps(ref)}, 5K prefill {pre} vs {pre0} tok/s); kept {out['pick']}")
            break
        if (r.get("kv_tokens") or 0) <= kv:
            notes.append(f"slot sweep: {n} slots gave no more context ({r.get('kv_tokens')} KV tokens vs {kv}); kept {out['pick']}")
            break
        pick, kv, out["pick"] = r, r["kv_tokens"], n
        if kv >= cap:
            break
    out["kv_tokens"] = kv
    if pick is not ref:
        out["row"] = pick
    return out


def ctx_stage(R: "Runner", inv: dict, win: dict, vis: bool, notes: list[str], first: dict | None = None, env: dict | None = None,
              cap: int | None = None) -> dict:
    """Stage 8: how much context fits on top of the winner. The engine sizes its KV at load from what VRAM is left
    (--max-seq-len-override is only a cap), so: load at the model's own maximum, read the KV tokens it allocated, send a prompt
    that fills it, and step down on a failure ("prompt is too long: N > M maximum", or a KV shortfall at load).
    2026-11-13: `first` is a good run of these args at `cap` (the as-stamped run), which stands in for the load here."""
    cap = cap or min(inv.get("context_length") or 262144, 524288) // 1024 * 1024
    ba = "fused" not in win["args"]
    out = {"cap": cap, "kv_tokens": None, "ctx": None, "ok": False, "tps": None, "fill_k": None}
    R.max_runs += 4  # a few extra runs: this stage is the point of a card's context
    try:
        n = (first.get("kv_tokens") or cap) if first and first.get("ok") else None
        for _ in range(0 if n else 3):  # load at the cap; a shortfall at load steps the cap down
            r = R.run("8-ctx", f"{win['label']}-ctx{cap // 1024}k", win["args"], tests="decode", vision=vis, env=env, base_args=ba, ctx=cap)
            if r is None:
                notes.append("context stage skipped: run budget spent")
                return out
            if r["ok"]:
                n = r.get("kv_tokens") or cap
                break
            notes.append(f"context {cap} failed to load: {r['error']}")
            cap = cap * 3 // 4 // 1024 * 1024
        if n is None:
            return out
        out["kv_tokens"] = n
        top = min(n, cap)
        out["ctx"] = top // 1024 * 1024  # what the KV holds, until a fill test says otherwise
        fill = top
        for _ in range(3):
            k = max(1, (fill - 1500) // 1000)
            r = R.run("8-ctx", f"{win['label']}-fill{k}k", win["args"], tests=f"decode,p{k}k", vision=vis, env=env, base_args=ba, ctx=cap)
            if r is None:
                notes.append("context fill test skipped: run budget spent")
                break
            out["fill_k"] = k
            if r["ok"]:
                out.update(ok=True, ctx=top // 1024 * 1024, tps=r["tps"].get(f"p{k}k"))
                return out
            mx = re.search(r"> (\d+) maximum", str(r.get("error")))
            nxt = int(mx[1]) if mx and int(mx[1]) < fill else fill * 85 // 100
            notes.append(f"{k}K prompt on a {fill}-token context failed: {err(r)}")
            fill = nxt // 1024 * 1024
            top = fill
            out["ctx"] = fill
        notes.append(f"context not confirmed by a fill test; {out['ctx']} is the KV size or the last step down")
    finally:
        R.max_runs -= 4
    return out


def load_flags() -> list[dict]:
    if not os.path.isfile(FLAGS_FILE):
        return []
    with open(FLAGS_FILE, encoding="utf-8") as f:
        return json.load(f)


# ---------- card ----------

def serve_base(launch: list[str]) -> str:
    """The probe's base args a hybrid launch line needs in front (a fused one needs none)."""
    if "fused" in launch:
        return ""
    return " --moe-cpu-layers auto" + ("" if "--moe-cache-size" in launch or "--moe-cache-auto" in launch else " --moe-cache-size 5000")


def advice(res: dict) -> list[str]:
    """2026-11-14: the card's 'What failed / try next / turned off' lines, keyed to what went wrong in this fit."""
    out: list[str] = []
    log, notes, floor, goal = res.get("log") or [], res.get("notes") or [], res.get("floor"), res.get("goal") or res.get("floor")
    fails = [r for r in log if not r.get("ok")]
    errs = [str(r.get("error")) for r in fails]
    # what the card's launch line already has: a tip naming one of those is left out, and a failure the fit got past is
    # worded as worked around
    have = set((res.get("confirm") or {}).get("args") or (res.get("use") or res.get("best") or {}).get("args") or [])
    loaded = res.get("best") is not None
    head = lambda what: f"**{'Worked around' if loaded else 'Failed'}: {what}**"
    tips = lambda opts: ", ".join(t for flag, t in opts if flag not in have) or "nothing left in this list"
    if res.get("best") is None:
        out.append("**Failed: nothing loaded.** " + (f"Last error: {err(fails[-1])[:300]}" if fails else "No run finished (run budget?)."))
    if any("available RAM" in e for e in errs):
        out.append(head("host RAM ran low") + " (the guard stopped a run). Try: " + tips([("", "close other apps"),
                   ("--mm-disable", "turn off vision (`--mm-disable vision`)"), ("--max-running-requests", "`--max-running-requests 1`")]) + ".")
    if any(r.get("kv_short") for r in fails):
        moe = res["inv"].get("experts")
        out.append(head("not enough VRAM for the KV cache") + " on some loads. Try: " + tips(
            ([("--moe-cache-size", f"fewer expert slots (`--moe-cache-size {min(3000, res['inv'].get('expert_slots') or 3000)}`)")] if moe else [("--embed-weights", "`--embed-weights host`"), ("", "a smaller quant")])
            + [("--max-running-requests", "`--max-running-requests 1`"), ("--cache-type", "`--cache-type naive`"), ("--kv-cache-dtype", "8-bit KV (lossy)")]) + ".")
    rooms = sorted({n for n in (slots_room(r) for r in fails) if n})
    if rooms:
        out.append(head("the 5000-slot expert cache did not fit") + f" (the engine had room for {', '.join(map(str, rooms))}); "
                   "the fit retries at that room. By hand: `--moe-cache-size` at or under it.")
    logs = [m[1] for e in errs for m in [re.search(r"exited \(15\).*?see (\S+)", e)] if m]
    if logs:
        out.append(head("the server died during load") + f" on {len(logs)} run(s) (exit 15). First log: `{logs[0]}`")
    spec_fail = [n for n in notes if re.match(r"(MTP|DFlash) k=\d+ failed", n)]
    if spec_fail and not res.get("spec"):
        out.append(f"**Turned off: spec decoding** ({spec_fail[0][:200]}). The card uses plain decode.")
    elif spec_fail:
        out.append(f"**Failed: a higher spec step** ({spec_fail[0][:160]}); the lower one is kept.")
    if any("vision does not fit" in n for n in notes):
        out.append("**Turned off: vision.** It does not fit with this config; `--mm-disable vision` is in the launch line. "
                   "Try: fewer expert slots or a lower ratio to make room, if you need it.")
    if any(n.startswith("tune flags dropped") for n in notes):
        out.append("**Turned off: the tune/class flags.** They slowed this model; the launch line runs without them.")
    confirm = res.get("confirm")
    if res.get("best") is not None and not res.get("tune") and not (confirm and confirm.get("ok")):
        out.append(f"**Failed: the as-stamped run** ({err(confirm)[:200]}); the card's numbers are ladder runs." if confirm
                   else "**Not run: the as-stamped run** (run budget spent); the card's numbers are ladder runs.")
    cm, k8 = res.get("ctxmax") or {}, res.get("kv8") or {}
    if floor and res.get("best") is not None and not res.get("tune") and cm.get("ctx") is None:
        out.append(f"**Not checked: context floor {floor}** (the context stage did not run). Rerun the fit with a bigger `--max-runs`.")
    elif floor and res.get("best") is not None and not res.get("tune") and ctx_verdict(res) == "under goal":
        more = tips([("--moe-cache-size", "fewer expert slots") if res["inv"].get("experts") else ("--embed-weights", "`--embed-weights host`"),
                     ("--max-running-requests", "`--max-running-requests 1`")])
        tries = ([f"the 8-bit KV line below ({k8.get('ctx')} tokens, lossy)"] if k8.get("launch") else ["8-bit KV (lossy; rerun with `FT_FIT_KV8=1`)"]) \
            + ([more] if more != "nothing left in this list" else []) + ["a smaller quant"]
        out.append(f"**Under goal: context.** {cm.get('ctx')} tokens fit: over the floor of {floor}, under the goal of {goal}. "
                   "The launch line is usable. For more context: " + ", ".join(tries) + ".")
    elif floor and res.get("best") is not None and not res.get("tune") and ctx_verdict(res) == "failed":
        tries = []
        if k8.get("launch"):
            tries.append(f"the 8-bit KV rescue line below ({k8.get('ctx')} tokens, lossy)")
        elif k8.get("ok"):
            tries.append("8-bit KV gave no real gain here")
        more = tips([("--moe-cache-size", "fewer expert slots") if res["inv"].get("experts") else ("--embed-weights", "`--embed-weights host`"),
                     ("--max-running-requests", "`--max-running-requests 1`")])
        tries += ([more] if more != "nothing left in this list" else []) + ["a smaller quant", f"or a lower floor (`--ctx-floor`, now {floor})"]
        out.append(f"**Failed: context floor.** {cm.get('ctx')} of {floor} tokens fit. Try: " + ", ".join(tries) + ".")
    return out

def ctx_verdict(res: dict) -> str | None:
    """2026-11-14: "met" (the goal holds), "under goal" (the floor holds, the goal does not), "failed" (under the floor),
    or None (no floor, or the context stage did not run). Both are capped at the model's own context length."""
    cm, floor = res.get("ctxmax") or {}, res.get("floor")
    if not floor or cm.get("ctx") is None:
        return None
    goal, cap = res.get("goal") or floor, cm.get("cap") or 1 << 30
    return "met" if cm["ctx"] >= min(goal, cap) else "under goal" if cm["ctx"] >= min(floor, cap) else "failed"


def card(res: dict) -> str:
    inv, best, spec, confirm = res["inv"], res["best"], res["spec"], res["confirm"]
    name = os.path.basename(inv["model"])
    L = [f"# Model card: {name}", "", f"Fitted {time.strftime('%Y-%m-%d %H:%M')} by `scripts/ft-fit.py` on RX 9070 16 GB / {inv['ram_total_gib']} GiB RAM, "
         + (f"every run at context {res['ctx']} (--max-seq-len-override)." if res.get("ctx") else "engine default context (no --ctx)."), ""]
    if best is None:
        L += ["**No config loaded.**", ""]
    elif res.get("tune"):
        L[0] = f"# Tune result: class {model_class(inv)} (run on {name})"
        L += [f"**Not a config to stamp.** Flags and the parallel report for every model of this class, measured on the plain config "
              f"`{' '.join(best['args'])}` ({tps(best)} tok/s). A fit of such a model applies the kept flags in its as-stamped run.", ""]
    else:
        win = res.get("use") or (spec if spec and better(spec, best) else best)
        is_spec = any(x in SPEC for x in win["args"])
        keep_env = {f["name"]: f["value"] for f in res["flags"] if f.get("keep") and f["kind"] == "env"}
        keep_args = [x for f in res["flags"] if f.get("keep") and f["kind"] == "arg" for x in f["value"].split()]
        vis_ok = confirm is not None and confirm.get("ok") and confirm.get("vision")
        launch = win["args"] + keep_args + ([] if vis_ok or not inv["mmproj"] else ["--mm-disable", "vision"])
        base = serve_base(launch)
        L += ["## Recommended launch", "", "```", f"ft serve --model-path \"{inv['model']}\"{base} {' '.join(launch)}", "```", ""]
        if res.get("reserve"):
            L += [f"Fitted with FREETOKEN_DESKTOP_RESERVE_GB={res['reserve']} on every run (VRAM held back for the desktop); serve with the same value.", ""]
        if keep_env:
            L += [f"Env: {' '.join(f'{k}={v}' for k, v in keep_env.items())}", ""]
        done = bool(confirm and confirm.get("ok"))
        L += [f"- Decode: **{tps(confirm) if done else tps(win)} tok/s** " + ("as stamped (this exact line in one run)" if done else "(ladder run; the as-stamped run did not pass)")
              + f"; ladder run of the same config {tps(win)} tok/s ({('DFlash' if 'dflash' in win['args'] else 'MTP') + ', not byte-identical to plain decode' if is_spec else 'plain decode'}); plain best {tps(best)} tok/s",
              f"- As-stamped run (decode + 5k prompt{', vision on' if vis_ok else ''}): " + (fmt(confirm) if confirm else "not run")]
        cm0 = res.get("ctxmax") or {}
        if res.get("floor"):
            v = ctx_verdict(res)
            L += [f"- Context goal {res.get('goal') or res['floor']} tokens, floor {res['floor']}: "
                  + {"met": "**met**", "under goal": "**under goal** (over the floor: usable)", "failed": "**NOT met** (under the floor)"}.get(v, "not checked")
                  + f" ({cm0.get('ctx')} tokens fit)"]
        L += [""]
    adv = advice(res)
    if adv:  # 2026-11-14: what failed, what to try, what was turned off, so a model with no card still gets a direction
        L += ["## What failed / try next / turned off", ""] + [f"- {x}" for x in adv] + [""]
    L += ["## Model", "", f"- {inv['arch']}, {inv['layers']} layers, {inv['experts']} experts ({inv['experts_used']} used), file {inv['file_gib']} GiB "
          f"(experts {inv['experts_gib']}, dense {inv['dense_gib']})", f"- Expert tensors: {inv['expert_types']}"
          + (f"; gate!=up layers {inv['gate_up_q8_layers']} load as Q8_0" if inv["gate_up_q8_layers"] else ""),
          f"- KV ~{(inv['kv_bytes_per_token'] or 0) // 1024} KiB/token (bf16 estimate): 32k context ~{inv['kv_gib_32k']} GiB",
          f"- Vision: {os.path.basename(inv['mmproj']) + ' (' + str(inv['mmproj_gib']) + ' GiB)' if inv['mmproj'] else 'none'}",
          f"- MTP: {'built-in' if inv['builtin_mtp'] else (', '.join(os.path.basename(s) for s in inv['sidecars']) + ('' if inv['sidecar_exact'] else ' (base model head, other expert count)') if inv['sidecars'] else 'none on disk')}",
          f"- Size-check verdict: {inv['verdict']} (host need {inv['host_need_text_gib']} GiB text, {inv['host_need_vision_gib']} with vision)", "",
          "## Runs", "", "| stage | config | result |", "|---|---|---|"]
    for r in res["log"]:
        L.append(f"| {r['stage']} | `{' '.join(r['args'])}`{' +vision' if r.get('vision') else ''}{' ' + str(r['env']) if r.get('env') else ''} ({r['tests']}) | {fmt(r)} |")
    cm = res.get("ctxmax")
    if cm and cm.get("ctx"):
        L += ["", "## Context", "", f"- {'Confirmed' if cm['ok'] else 'Unconfirmed'} context on top of the winner: **{cm['ctx']} tokens** "
              f"(engine KV {cm['kv_tokens']} tokens; model max {cm['cap']}); fill test {cm.get('fill_k')}K prompt"
              + (f", {cm['tps']} tok/s decode at that depth" if cm.get("tps") else "")]
    k8 = res.get("kv8")
    if k8:  # 2026-11-13: a report only; 2026-11-14: run only as a floor rescue, with its own launch line when it holds more
        L += ["", "## 8-bit KV (lossy, not in the launch line)", ""]
        if k8.get("ok"):
            L += [f"- `--kv-cache-dtype {k8['format']}` on the as-stamped config: KV **{k8.get('kv_tokens')} tokens** (16-bit: {k8.get('base_kv_tokens')}), "
                  f"decode {k8.get('decode')} tok/s (16-bit: {k8.get('base_decode')}); decode-test text {'identical' if k8.get('text_same') else 'worded differently'} "
                  "(an exact-text check: different wording is expected and says nothing about accuracy)."]
            if k8.get("launch"):
                L += [f"- **Rescue line**, if you need the context more than the accuracy: {k8.get('ctx')} tokens "
                      + ("(confirmed by a fill test)" if k8.get("ctx_ok") else "(NOT confirmed by a fill test)")
                      + (", meets the goal" if k8.get("goal_ok") else ", meets the floor" if k8.get("floor_ok") else ", still under the floor") + ":", "",
                      "```", f"ft serve --model-path \"{inv['model']}\"{serve_base(k8['launch'])} {' '.join(k8['launch'])}", "```", ""]
            L += ["- Measured cost (ctx_rot, 35B distill, 2026-11-14): 8K 7/10 vs 9/10 at 16-bit (paraphrase decoys and a count), "
                  "and one hang in a 30K prefill. Use it when the 16-bit context is too short, not by default."]
        else:
            L += [f"- `--kv-cache-dtype {k8['format']}` did not run: {k8.get('error') or 'skipped (run budget)'}"]
    sl = res.get("slots")
    if sl and sl.get("rows") and sl.get("mode") == "auto":  # 2026-11-12
        L += ["", "## Expert slots against context", "", f"--slots-mode auto: the engine sized the slots itself (--moe-cache-auto, KV floor {sl['kv_reserve']} tokens): "
              f"**{sl['pick']} slots**" + ("" if sl["rows"][0]["ok"] and sl["rows"][0]["slots"] else " (its pick was not found, so the 5000-slot config stays)")
              + f". Loaded at context {sl['cap']}.", "", "| slots | decode tok/s | 5K prefill tok/s | KV tokens | max VRAM GiB |", "|---|---|---|---|---|"]
        L += [f"| {x['slots'] or 'auto'} | {x['decode'] if x['ok'] else 'FAIL'} | {x['prefill'] or ''} | {x['kv_tokens'] or ''} | {x['max_vram_gib'] or ''} |" for x in sl["rows"]]
    elif sl and sl.get("rows"):
        rule = (f"the fastest count that holds the goal of {sl['floor']} KV tokens" if sl.get("floor") else
                f"lowest count within {int(SLOT_DECODE * 100)}% decode and {int(SLOT_PREFILL * 100)}% 5K prefill of 5000, while the KV still grew")
        L += ["", "## Expert slots against context", "", f"Picked **{sl['pick']} slots** ({rule}). Loaded at context {sl['cap']}.", "", "| slots | decode tok/s | 5K prefill tok/s | KV tokens | max VRAM GiB |", "|---|---|---|---|---|"]
        L += [f"| {x['slots']} | {x['decode'] if x['ok'] else 'FAIL'} | {x['prefill'] or ''} | {x['kv_tokens'] or ''} | {x['max_vram_gib'] or ''} |" for x in sl["rows"]]
    tf = res.get("tune_from")
    if tf:
        L += ["", f"Flags below come from the tune run of {tf.get('when')} on {os.path.basename(str(tf.get('model')))} (class {tf.get('class')}); this fit only ran them in the as-stamped run."]
    if res["flags"]:
        L += ["", "## Flags", ""] + [f"- `{f['id']}` ({f['status']}, lossless {f.get('lossless')}): {f['result']}{' **kept**' if f.get('keep') else ''}" for f in res["flags"]]
    par = res.get("parallel")
    if par and par.get("rows"):  # 2026-11-12 stage 6p: a report, the launch line above is for one request
        L += ["", "## Parallel requests", "", f"One request decodes at {par['plain_tps']} tok/s. Best: **{par['best_n']} at once**"
              + ("" if par["best_n"] > 1 else " (more requests gave no more total speed)") + ". Not in the launch line: the context per request "
              "is your call. To use N: `--max-running-requests N --cuda-graph-max-bs N` and FT_GRAPH_MAX_BS=N.", "",
              "| requests | graph bs | extra args | total tok/s | per request tok/s | one request tok/s | KV tokens | max VRAM GiB | min RAM GiB |",
              "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {x['n']} | {x.get('graph_bs', x['n'])} | {x.get('extra') or ''} | {x['total_tps'] if x['ok'] else 'FAIL: ' + str(x['error'])[:80]} | {' '.join(str(v) for v in x['per_req_tps'] or [])} | {x['single_tps'] or ''} "
              f"| {x['kv_tokens'] or ''} | {x['max_vram_gib'] or ''} | {x['min_ram_gib'] or ''} |" for x in par["rows"]]
    if res["notes"]:
        L += ["", "## Notes", ""] + [f"- {n}" for n in res["notes"]]
    return "\n".join(L) + "\n"


SPEC = ("--speculative-algorithm", "--speculative-num-steps", "--speculative-draft-model-path")


def winner(res: dict) -> dict:
    """The card's fields from a ladder: OpenFT's {args, cache, head, env} plus the numbers behind them. args leaves out the
    context (a card has its own ctx), and the head and expert cache when OpenFT can add them from the card's head and cache."""
    inv, best, spec, confirm = res["inv"], res["best"], res["spec"], res["confirm"]
    out = {"model": inv["model"], "tag": res.get("tag"), "ctx": res.get("ctx") or None, "verdict": inv["verdict"],
           "when": time.strftime("%Y-%m-%d %H:%M"), "runs": res.get("runs", 0), "reused": res.get("reused", 0),
           "notes": res["notes"], "ok": best is not None, "advice": advice(res)}
    if best is None:
        return out
    if res.get("tune"):  # 2026-11-13: a tune run has no config to stamp; OpenFT and the bench skip the card for it
        out.update(tune=True, tune_class=model_class(inv), plain_tps=tps(best), parallel=res.get("parallel"),
                   flags=[{k: f.get(k) for k in ("id", "kind", "name", "value", "keep", "result")} for f in res["flags"]])
        return out
    win = res.get("use") or (spec if spec and better(spec, best) else best)
    is_spec = any(x in SPEC for x in win["args"])
    done = bool(confirm and confirm.get("ok"))
    keep_env = {f["name"]: f["value"] for f in res["flags"] if f.get("keep") and f["kind"] == "env"}
    keep_args = [x for f in res["flags"] if f.get("keep") and f["kind"] == "arg" for x in f["value"].split()]
    vis_ok = bool(confirm and confirm.get("ok") and confirm.get("vision"))
    fused = "fused" in win["args"]
    a, plain, spec_args, head = list(win["args"]), [], [], None
    i, picked = 0, (0 if "--moe-cache-auto" in a else 5000)  # auto + reserve (2026-11-13): no fixed count for the bench to add
    while i < len(a):  # split the spec args off
        if a[i] in SPEC:
            spec_args += a[i:i + 2]
            head = head or {"kind": "mtp", "path": "", "steps": 3}
            if a[i] == "--speculative-num-steps":
                head["steps"] = int(a[i + 1])
            elif a[i] == "--speculative-draft-model-path":
                head["path"] = a[i + 1]
            else:
                head["kind"] = a[i + 1]
            i += 2
        elif a[i] == "--moe-cache-size":  # the slot sweep's pick: it goes in the card's cache field, not its args
            picked = int(a[i + 1])
            i += 2
        else:
            plain.append(a[i])
            i += 1
    cache = 0 if fused else picked  # 5000 = the probe's base args on every hybrid run: --moe-cpu-layers auto --moe-cache-size 5000
    args = plain if fused else ["--moe-cpu-layers", "auto"] + plain
    if head and not cache:  # OpenFT refuses a head on a card with no expert cache: keep the spec args as plain args
        args, head = args + spec_args, None
    args += keep_args + ([] if vis_ok or not inv["mmproj"] else ["--mm-disable", "vision"])
    ram = [r["min_ram_gib"] for r in (win, confirm) if r and r.get("min_ram_gib") is not None]
    vram = [r["max_vram_gib"] for r in (win, confirm) if r and r.get("max_vram_gib") is not None]
    # tps is the as-stamped run's (the whole launch line in one process); ladder_tps is the same config's earlier ladder run
    out.update(probe_env=R.probe_env)  # 2026-11-14: the engine env under every run, so a bench can tell if it serves the same
    out.update(args=" ".join(args), cache=cache, head=head, env=keep_env, pick="spec" if is_spec else "plain",
               tps=tps(confirm) if done else tps(win), final_tps=tps(confirm) if done else None, ladder_tps=tps(win),
               plain_tps=tps(best), spec_tps=tps(spec) if spec else None,
               p5k_tps=confirm.get("tps", {}).get("p5k") if done else None,
               confirm_ok=done, vision=bool(inv["mmproj"]), vision_ok=vis_ok,
               min_ram_gib=min(ram) if ram else None, max_vram_gib=max(vram) if vram else None, load_s=(confirm if done else win).get("load_s"),
               text_same=bool(not is_spec or win.get("text") == best.get("text")),
               flags=[f["id"] for f in res["flags"] if f.get("keep")], tune_from=res.get("tune_from"), kv8=res.get("kv8"))
    if res.get("floor"):
        cm0 = res.get("ctxmax") or {}
        g = res.get("goal") or res["floor"]
        out.update(ctx_floor=res["floor"], floor_ok=(cm0.get("ctx") or 0) >= min(res["floor"], cm0.get("cap") or res["floor"]),
                   ctx_goal=g, goal_ok=(cm0.get("ctx") or 0) >= min(g, cm0.get("cap") or g), ctx_verdict=ctx_verdict(res))
    sl = res.get("slots")
    if sl and sl.get("rows"):
        out.update(slots=sl["pick"], slot_rows=sl["rows"], slots_mode=sl.get("mode", "sweep"))
    if res.get("parallel"):  # 2026-11-12: reported only; args above are for one request
        out["parallel"] = res["parallel"]
    if res.get("reserve"):
        out.update(reserve=res["reserve"], env=dict(keep_env, FREETOKEN_DESKTOP_RESERVE_GB=str(res["reserve"])))
    cm = res.get("ctxmax")
    if cm and cm.get("ctx"):  # the card's context is what fits on top of the winner, not the ladder's 64K
        out.update(ctx_fit=out["ctx"], ctx=cm["ctx"], kv_tokens=cm["kv_tokens"], ctx_ok=cm["ok"], ctx_fill_tps=cm.get("tps"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="+")
    ap.add_argument("--plan-only", action="store_true", help="stages 1-2 only, no GPU")
    ap.add_argument("--max-runs", type=int, default=10)
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument("--vram-gib", type=float, default=16.0)
    ap.add_argument("--desktop-gib", type=float, default=1.2, help="VRAM the desktop holds (guard base_vram 1.16-1.37 on 11-07)")
    ap.add_argument("--extra-gib", type=float, default=5.0, help="host RAM of the serve process beyond the expert banks: 11-07 sweep measured 2.4-7.8 GiB over the expert bytes (free-phys counter, an upper bound)")
    ap.add_argument("--json", action="store_true", help="--plan-only: print the inventory as JSON")
    ap.add_argument("--ctx", type=int, default=0, help="run every stage at this context (--max-seq-len-override), so the card's context is the one measured")
    ap.add_argument("--tag", help="name for the resume file and card (default: from the file name)")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier runs in speedprobe/fit/<tag>.jsonl (they stay in the file)")
    ap.add_argument("--no-slots", action="store_true", help="skip stage 4s, the expert-slot sweep (the card keeps 5000 slots)")
    ap.add_argument("--slots-mode", choices=("sweep", "auto"), default="sweep", help="stage 4s: sweep = 5000/4000/3000/2000 slots (default); auto = one --moe-cache-auto run, the engine picks")
    ap.add_argument("--kv-reserve", type=int, default=0, help="--slots-mode auto: KV tokens held back before the experts fill VRAM (--kv-reserve-tokens). Default 0: --ctx, or 65536")
    ap.add_argument("--dflash-head", help="a DFlash draft checkpoint: stage 5 tries it after MTP and keeps the faster. Default: none (no disk search)")
    ap.add_argument("--no-parallel", action="store_true", help="skip stage 6p, the 2 and 4 parallel-request runs (a report only)")
    ap.add_argument("--reserve-gib", default=os.environ.get("FT_FIT_RESERVE_GB", ""), help="FREETOKEN_DESKTOP_RESERVE_GB on every run: VRAM held back for the desktop (GiB, or auto). Default: none")
    ap.add_argument("--no-preflight", action="store_true", help="run the GPU loads even when the pre-flight check (engine arch list, tensor types) says stop")
    ap.add_argument("--ctx-goal", type=int, default=0, help="the context to aim for (tokens): slots are picked as the fastest count whose KV reaches it. Default 0: --ctx, or 65536")
    ap.add_argument("--ctx-floor", type=int, default=0, help="the least context that still passes (tokens); between it and the goal the card says under goal. "
                    "Default 0: 32768 (or the goal, if lower). A floor over the goal raises the goal")
    ap.add_argument("--tune", action="store_true", help="tune run: flags (winners run twice) and the parallel report on the plain config, saved to speedprobe/tune/<class>.json "
                    "for every later fit of that model class. Writes no config to stamp")
    ap.add_argument("--token-map", default="auto", help="stage 5: --speculative-token-map for the MTP winner. auto (default) = speedprobe/token-maps/<model>.bin, written once "
                    "by scripts/make_token_map.py (tokenizer only, no GPU); off; or a file")
    ap.add_argument("--no-kv8", action="store_true", help="skip the 8-bit KV row (one run, a report only; never in the launch line)")
    ap.add_argument("--result", help="write the winner as JSON here (the bench's ftfit test reads it); one model only")
    a = ap.parse_args()
    if (a.result or a.tag) and len(a.model) > 1:
        ap.error("--result and --tag take one model")
    for m in a.model:
        if os.path.basename(m).lower().startswith("mmproj"):
            print(f"== {os.path.basename(m)}: a vision encoder, not a model; skipped (it is found next to the model)")
            continue
        inv = inventory(m, a.vram_gib, a.desktop_gib, a.extra_gib)
        if a.json:
            print(json.dumps(inv))
        else:
            print_inventory(inv)
        pf = inv["preflight"] = preflight(inv)
        for w in pf["warn"]:
            print("  pre-flight:", w)
        for w in pf["stop"]:
            print("  PRE-FLIGHT STOP:", w)
        if pf["stop"] and not a.no_preflight:
            inv["verdict"] = "unsupported"
        if a.plan_only or inv["verdict"] in ("no-fit", "unsupported"):
            if not a.plan_only:
                print("  stopping: " + ("the pre-flight check says it cannot load (--no-preflight runs it anyway)" if inv["verdict"] == "unsupported"
                                        else "the size check says it cannot fit"))
            if a.result:  # no runs: the size check alone
                with open(a.result, "w", encoding="utf-8") as f:
                    json.dump({"model": m, "tag": a.tag or slug(m), "ctx": a.ctx or None, "verdict": inv["verdict"], "ok": False,
                               "plan_only": a.plan_only, "runs": 0, "reused": 0, "notes": [], "inv": inv}, f, indent=1)
            continue
        if inv["verdict"].startswith("tight"):
            print(f"  warning: host RAM is tight ({inv['ram_avail_gib']} GiB free); the guard will stop a run before it runs out")
        res = ladder(inv, a)
        res.update(tag=a.tag or slug(m), ctx=res.pop("ctx_used", a.ctx))
        write_out(res, a, m)


def write_out(res: dict, a, m: str):
    os.makedirs(os.path.join(PROBE_OUT, "cards"), exist_ok=True)
    cp = os.path.join(PROBE_OUT, "cards", f"{a.tag or slug(m)}{'-tune' if res.get('tune') else ''}.md")  # a tune run never overwrites a model's card
    stopped = RESETS["n"] >= GPU_RESET_STOP
    if stopped:  # 2026-11-14: a fit cut short by GPU resets is not a fit: the bench must not take its partial winner as the newest good one
        res.setdefault("notes", []).append(f"STOPPED after {RESETS['n']} Windows GPU resets: this card is partial, not a fit result. Re-fit once the PC is checked.")
    with open(cp, "w", encoding="utf-8") as f:
        f.write(card(res))
    print(f"  card: {cp}")
    if a.result:
        keys = ("stage", "label", "args", "tests", "vision", "env", "ok", "tps", "error", "min_ram_gib", "max_vram_gib", "load_s", "when")
        w = dict(winner(res), card=cp, log=[{k: r.get(k) for k in keys} for r in res["log"]])
        if RESETS["n"]: w["gpu_resets"] = RESETS["n"]
        if stopped: w["ok"] = False
        with open(a.result, "w", encoding="utf-8") as f:
            json.dump(w, f, indent=1)
        print(f"  result: {a.result}")


if __name__ == "__main__":
    main()
