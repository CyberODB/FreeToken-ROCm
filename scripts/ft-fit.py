"""Fit a GGUF to this PC and recommend a launch config (a model card). Plan: T:\\robot\\docs\\19-freetoken-fit-flow-plan.md

  py scripts/ft-fit.py <model.gguf> --plan-only        size check + upgrades, no GPU
  py scripts/ft-fit.py <model.gguf> [--max-runs 10]    guarded run ladder, then speedprobe/cards/<model>.md

Stages: 1 inventory/size check, 2 upgrades (MTP), 3 strategy, 4 memory ratio, 4s expert slots against context
(2026-11-11; --no-slots skips it; undo: ft-fit.py.bak-20261111-slots), 5 spec decoding, 6 flags,
7 fit confirmation with vision + a 5k prompt, 8 context, then the card. Every GPU run is gpu-guard.py around ft-speedprobe.py.
2026-11-12 (undo: ft-fit.py.bak-20261112-stages):
  4s  --slots-mode auto: one --moe-cache-auto run (KV floor --kv-reserve) instead of the sweep; the engine's slot count
      is read from its log and becomes the card's cache.
  5   --dflash-head <path>: a DFlash draft is tried after MTP (k 2-4); the faster of the two is the spec config.
  6   a fit-flags.json entry may carry "values" (one run each), "tests", "metric" (decode | prefill:<test> | total:<test>)
      and "lossless"; kept only when lossless is true, the text is identical, the metric gains over 2% and nothing else
      in the run is over 5% worse. A tests string other than "decode" gets its own plain reference run.
  6p  parallel requests (2 and 4 at once, test cN): report only, never changes the winner; --no-parallel skips it.
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
# 2026-11-11 slot sweep (stage 4s): expert GPU slots traded for KV (context). 5000 is the probe's base; each step down is kept
# while decode stays within SLOT_DECODE and the 5K prefill within SLOT_PREFILL of the 5000 run, and the KV keeps growing.
SLOTS = (4000, 3000, 2000)
SLOT_DECODE, SLOT_PREFILL = 0.05, 0.10
FLAG_WORSE = 0.05  # 2026-11-12 stage 6: a flag is not kept when another number of its run drops more than this
PARALLEL = (2, 4)  # 2026-11-12 stage 6p: requests at once


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
    odd = [L for L in range(int(g("block_count", 0) or 0))
           if f"blk.{L}.ffn_gate_exps.weight" in by and by[f"blk.{L}.ffn_gate_exps.weight"] != by.get(f"blk.{L}.ffn_up_exps.weight")]
    # KV per token (bf16 assumed): only full-attention layers hold KV in qwen35*/hybrid archs
    layers, kvh, kl, vl = g("block_count"), g("attention.head_count_kv"), g("attention.key_length"), g("attention.value_length")
    kv_tok = None
    if layers and kvh and kl and vl and not isinstance(kvh, (list, tuple)):
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
        "expert_types": types, "gate_up_q8_layers": odd, "kv_bytes_per_token": kv_tok,
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
                    r = json.loads(line)
                    self.done[r["key"]] = r
        self.base_env: dict = {}  # env on every run (the desktop VRAM reserve)
        self.runs = 0
        self.reused = 0
        self.log: list[dict] = []

    def run(self, stage: str, label: str, args: list[str], tests: str = "decode", vision: bool = False,
            env: dict | None = None, base_args: bool = True, ctx: int | None = None) -> dict | None:
        """One guarded probe run; None when the run budget is spent."""
        name = f"fit-{self.tag}-{label}"[:90]
        ctx = self.ctx if ctx is None else ctx
        env = {**self.base_env, **(env or {})} or None
        key =json.dumps([name, args, tests, vision, env or {}, base_args] + ([ctx] if ctx else []))
        if key in self.done:
            r = self.done[key]
            self.reused += 1
            print(f"  [{stage}] {label}: reused  {fmt(r)}")
            self.log.append(r)
            return r
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
        except OSError:
            r["kv_tokens"] = None
            r["kv_short"] = False
            r["slots_auto"] = None
        with open(self.state_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        print(f"      -> {fmt(r)}")
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


# ---------- ladder ----------

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
            if not r["ok"]:
                notes.append(f"{strat} at ratio {ratio:.2f} failed: {err(r)}")
                if kv_short(r) and strat == "hybrid" and last is None:
                    continue  # hybrid starts at the lowest ratio: a higher one leaves more room for KV
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
    if best is None:
        notes.append("no config loaded; see the errors above")
        return {"inv": inv, "ctx_used": R.ctx, "best": None, "spec": None, "flags": [], "confirm": None, "notes": notes, "log": R.log, "runs": R.runs, "reused": R.reused}
    plain = best
    slots = None
    if inv.get("experts") and "hybrid" in best["args"] and not a.no_slots:
        if getattr(a, "slots_mode", "sweep") == "auto":  # 2026-11-12: one --moe-cache-auto run instead of the sweep
            slots = slot_auto_stage(R, inv, best, notes, int(getattr(a, "kv_reserve", 0) or a.ctx or 65536))
        else:
            slots = slot_stage(R, inv, best, notes)
        if slots.get("row"):
            plain = slots.pop("row")
    # stage 5: spec decoding. MTP first; 2026-11-12: then a DFlash draft when --dflash-head names one. The faster is `spec`.
    spec = None

    def spec_try(kind: str, extra: list[str]) -> dict | None:
        top = None
        for k in (2, 3, 4):
            r = R.run("5-spec", f"{plain['label']}-{kind}{k}", plain["args"] + ["--speculative-algorithm", kind, "--speculative-num-steps", str(k)] + extra)
            if r is None or not r["ok"]:
                if r is not None:
                    notes.append(f"{'MTP' if kind == 'mtp' else 'DFlash'} k={k} failed: {err(r)}")
                break
            if top is None or better(r, top):
                top = r
            elif tps(r) < tps(top):
                break
        return top

    draft = [] if inv["builtin_mtp"] else (["--speculative-draft-model-path", inv["sidecars"][0]] if inv["sidecars"] else None)
    if draft is None:
        notes.append("no MTP draft head on disk; " + ("MTP skipped" if getattr(a, "dflash_head", None) else "spec decoding skipped"))
    else:
        spec = spec_try("mtp", draft)
    dhead = getattr(a, "dflash_head", None)
    if dhead and not os.path.exists(dhead):
        notes.append(f"--dflash-head {dhead} does not exist; DFlash skipped")
    elif dhead:
        R.max_runs += 3  # its own runs: the later stages keep their budget
        dfl = spec_try("dflash", ["--speculative-draft-model-path", dhead])
        if dfl is not None and spec is not None:
            notes.append(f"spec decoding: MTP {tps(spec)} tok/s, DFlash {tps(dfl)} tok/s; {'DFlash' if better(dfl, spec) else 'MTP'} kept")
        if dfl is not None and (spec is None or better(dfl, spec)):
            spec = dfl
    # stage 6: flags, on the plain config (lossless = byte-identical decode text)
    flags_out = flag_stage(R, inv, plain)
    # stage 6p (2026-11-12): parallel requests on the plain config; a report only, the winner's args stay as they are
    par = None if getattr(a, "no_parallel", False) else parallel_stage(R, plain, notes)
    # stage 7: fit confirmation (vision if present, 5k prompt)
    vis = bool(inv["mmproj"])
    confirm = None
    args = list(plain["args"])
    for attempt in range(2):
        confirm = R.run("7-confirm", f"{plain['label']}-confirm{attempt}{'-vis' if vis else ''}", args, tests="decode,p5k", vision=vis)
        if confirm is None or confirm["ok"]:
            break
        notes.append(f"confirm {'with vision ' if vis else ''}at {' '.join(args)} failed: {err(confirm)}")
        m = [i for i, x in enumerate(args) if x == "--memory-ratio"]
        if not m:
            break
        args[m[0] + 1] = f"{float(args[m[0] + 1]) - 0.05:.2f}"
    if vis and confirm is not None and not confirm["ok"]:
        confirm = R.run("7-confirm", f"{plain['label']}-confirm-novis", plain["args"], tests="decode,p5k")
        notes.append("vision does not fit with this config: launch with --mm-disable vision")
    ctxmax = ctx_stage(R, inv, spec if spec and better(spec, plain) else plain, vis and bool(confirm and confirm.get("ok")), notes)
    return {"inv": inv, "ctx_used": R.ctx, "best": plain, "spec": spec, "flags": flags_out, "confirm": confirm, "ctxmax": ctxmax, "slots": slots,
            "parallel": par, "reserve": a.reserve_gib or None, "notes": notes, "log": R.log, "runs": R.runs, "reused": R.reused}


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
        for v in values(fl):
            label = f"{plain['label']}-{fid}" + ("-" + re.sub(r"[^a-z0-9.]+", "", (v.split() or ["x"])[-1].lower()) if fl.get("values") else "")
            r = R.run("6-flag", label, plain["args"] + ([] if fl["kind"] == "env" else v.split()), tests=tests,
                      env={fl["name"]: v} if fl["kind"] == "env" else None)
            if r is None:
                spent = True
                break
            ok = bool(r.get("ok"))
            same, m = ok and r.get("text") == plain.get("text"), flag_metric(r, metric)
            checks = [("decode", tps(r), tps(ref)), ("5K prefill", (r.get("prefill") or {}).get("p5k"), (ref.get("prefill") or {}).get("p5k"))]
            worse = [f"{n} {new} vs {old}" for n, new, old in checks if ok and new and old and new < old * (1 - FLAG_WORSE)]
            rows.append({"label": label, "value": v, "ok": ok, "metric": m if ok else None, "ref": ref_m, "text_same": same, "worse": worse,
                         "pass": bool(fl.get("lossless") is True and same and ref_m and m > ref_m * (1 + SAME) and not worse)})
            said.append((f"`{v}`: " if fl.get("values") else "") + fmt(r) + ("" if metric == "decode" or not ok else f"; {metric} {m} vs {ref_m}")
                        + ("; text identical" if same else ("; TEXT DIFFERS" if ok else "")) + (f"; worse: {', '.join(worse)}" if worse else ""))
        if rows:
            top = max(rows, key=lambda x: (x["pass"], x["ok"], x["metric"] or 0))
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
    out = {"rows": [], "best_n": 1, "plain_tps": tps(plain)}
    R.max_runs += len(PARALLEL)
    base = list(plain["args"])
    while "--max-running-requests" in base:  # a dense winner may carry --max-running-requests 1
        i = base.index("--max-running-requests")
        del base[i:i + 2]
    for n in PARALLEL:
        t = f"c{n}"
        r = R.run("6p-parallel", f"{plain['label']}-par{n}", base + ["--max-running-requests", str(n), "--cuda-graph-max-bs", str(n)],
                  tests=f"decode,{t}", env={"FT_GRAPH_MAX_BS": str(n)})
        if r is None:
            notes.append(f"parallel stage: {n} requests skipped, run budget spent")
            break
        ok = bool(r.get("ok"))
        out["rows"].append({"n": n, "ok": ok, "total_tps": (r.get("tps") or {}).get(t) if ok else None, "per_req_tps": (r.get("per_req") or {}).get(t),
                            "single_tps": tps(r) if ok else None, "kv_tokens": r.get("kv_tokens"), "max_vram_gib": r.get("max_vram_gib"),
                            "min_ram_gib": r.get("min_ram_gib"), "error": None if ok else str(r.get("error"))[:200]})
        if not ok:
            notes.append(f"parallel stage: {n} requests at once failed: {err(r)}")
            break
    good = [x for x in out["rows"] if x["ok"] and x["total_tps"]]
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
    r = R.run("4s-slots", f"{win['label']}-sauto", ["--moe-cpu-layers", "auto"] + win["args"] + ["--moe-cache-auto", "--kv-reserve-tokens", str(reserve)],
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
        out.update(pick=n, kv_tokens=r.get("kv_tokens"), row=dict(r, args=win["args"] + ["--moe-cache-size", str(n)]))
    return out


def slot_stage(R: "Runner", inv: dict, win: dict, notes: list[str]) -> dict:
    """Stage 4s (2026-11-11): the expert GPU slots against the KV (context) on the ratio winner. Fewer slots leave VRAM for
    KV, so the fit keeps the lowest slot count whose decode and 5K prefill stay within SLOT_DECODE / SLOT_PREFILL of the
    5000-slot run. Runs load at the model's own maximum context, so "Allocating N tokens for KV cache" is the real room and
    not the ladder's cap. Returns the table, the pick, and (when the pick is not 5000) its row, which becomes the plain winner."""
    cap = min(inv.get("context_length") or 262144, 524288) // 1024 * 1024
    out = {"ref": 5000, "pick": 5000, "rows": [], "cap": cap}
    R.max_runs += len(SLOTS) + 1

    def one(n: int) -> dict | None:
        r = R.run("4s-slots", f"{win['label']}-s{n}", win["args"] + (["--moe-cache-size", str(n)] if n != 5000 else []), tests="decode,p5k", ctx=cap)
        if r is not None:
            out["rows"].append({"slots": n, "ok": r["ok"], "decode": tps(r), "prefill": (r.get("prefill") or {}).get("p5k"),
                                "kv_tokens": r.get("kv_tokens"), "max_vram_gib": r.get("max_vram_gib"), "error": None if r["ok"] else str(r.get("error"))[:200]})
        return r

    ref = one(5000)
    if ref is None or not ref["ok"]:
        notes.append("slot sweep skipped: " + ("run budget spent" if ref is None else f"the 5000-slot run at context {cap} failed: {err(ref)}"))
        return out
    pick, kv = ref, ref.get("kv_tokens") or 0
    if kv >= cap:
        notes.append(f"slot sweep: 5000 slots already hold the model's whole context ({cap}); nothing to trade")
        return out
    pre0 = (ref.get("prefill") or {}).get("p5k") or 0
    for n in SLOTS:
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


def ctx_stage(R: "Runner", inv: dict, win: dict, vis: bool, notes: list[str]) -> dict:
    """Stage 8: how much context fits on top of the winner. The engine sizes its KV at load from what VRAM is left
    (--max-seq-len-override is only a cap), so: load at the model's own maximum, read the KV tokens it allocated, send a prompt
    that fills it, and step down on a failure ("prompt is too long: N > M maximum", or a KV shortfall at load)."""
    cap = min(inv.get("context_length") or 262144, 524288) // 1024 * 1024
    out = {"cap": cap, "kv_tokens": None, "ctx": None, "ok": False, "tps": None, "fill_k": None}
    R.max_runs += 4  # a few extra runs: this stage is the point of a card's context
    try:
        n = None
        for _ in range(3):  # load at the cap; a shortfall at load steps the cap down
            r = R.run("8-ctx", f"{win['label']}-ctx{cap // 1024}k", win["args"], tests="decode", vision=vis, ctx=cap)
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
            r = R.run("8-ctx", f"{win['label']}-fill{k}k", win["args"], tests=f"decode,p{k}k", vision=vis, ctx=cap)
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

def card(res: dict) -> str:
    inv, best, spec, confirm = res["inv"], res["best"], res["spec"], res["confirm"]
    name = os.path.basename(inv["model"])
    L = [f"# Model card: {name}", "", f"Fitted {time.strftime('%Y-%m-%d %H:%M')} by `scripts/ft-fit.py` on RX 9070 16 GB / {inv['ram_total_gib']} GiB RAM, "
         + (f"every run at context {res['ctx']} (--max-seq-len-override)." if res.get("ctx") else "engine default context (no --ctx)."), ""]
    if best is None:
        L += ["**No config loaded.**", ""]
    else:
        win = spec if spec and better(spec, best) else best
        keep_env = {f["name"]: f["value"] for f in res["flags"] if f.get("keep") and f["kind"] == "env"}
        keep_args = [x for f in res["flags"] if f.get("keep") and f["kind"] == "arg" for x in f["value"].split()]
        vis_ok = confirm is not None and confirm.get("ok") and confirm.get("vision")
        launch = win["args"] + keep_args + ([] if vis_ok or not inv["mmproj"] else ["--mm-disable", "vision"])
        base = "" if "fused" in win["args"] else " --moe-cpu-layers auto" + ("" if "--moe-cache-size" in launch else " --moe-cache-size 5000")
        L += ["## Recommended launch", "", "```", f"ft serve --model-path \"{inv['model']}\"{base} {' '.join(launch)}", "```", ""]
        if res.get("reserve"):
            L += [f"Fitted with FREETOKEN_DESKTOP_RESERVE_GB={res['reserve']} on every run (VRAM held back for the desktop); serve with the same value.", ""]
        if keep_env:
            L += [f"Env: {' '.join(f'{k}={v}' for k, v in keep_env.items())}", ""]
        L += [f"- Decode: **{tps(win)} tok/s** ({('DFlash' if 'dflash' in win['args'] else 'MTP') + ', not byte-identical to plain decode' if win is spec else 'plain decode'}); plain best {tps(best)} tok/s",
              f"- Fit check (decode + 5k prompt{', vision on' if vis_ok else ''}): " + (fmt(confirm) if confirm else "not run"), ""]
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
    sl = res.get("slots")
    if sl and sl.get("rows") and sl.get("mode") == "auto":  # 2026-11-12
        L += ["", "## Expert slots against context", "", f"--slots-mode auto: the engine sized the slots itself (--moe-cache-auto, KV floor {sl['kv_reserve']} tokens): "
              f"**{sl['pick']} slots**" + ("" if sl["rows"][0]["ok"] and sl["rows"][0]["slots"] else " (its pick was not found, so the 5000-slot config stays)")
              + f". Loaded at context {sl['cap']}.", "", "| slots | decode tok/s | 5K prefill tok/s | KV tokens | max VRAM GiB |", "|---|---|---|---|---|"]
        L += [f"| {x['slots'] or 'auto'} | {x['decode'] if x['ok'] else 'FAIL'} | {x['prefill'] or ''} | {x['kv_tokens'] or ''} | {x['max_vram_gib'] or ''} |" for x in sl["rows"]]
    elif sl and sl.get("rows"):
        L += ["", "## Expert slots against context", "", f"Picked **{sl['pick']} slots** (lowest count within {int(SLOT_DECODE * 100)}% decode and {int(SLOT_PREFILL * 100)}% 5K prefill of 5000, "
              f"while the KV still grew). Loaded at context {sl['cap']}.", "", "| slots | decode tok/s | 5K prefill tok/s | KV tokens | max VRAM GiB |", "|---|---|---|---|---|"]
        L += [f"| {x['slots']} | {x['decode'] if x['ok'] else 'FAIL'} | {x['prefill'] or ''} | {x['kv_tokens'] or ''} | {x['max_vram_gib'] or ''} |" for x in sl["rows"]]
    if res["flags"]:
        L += ["", "## Flags", ""] + [f"- `{f['id']}` ({f['status']}, lossless {f.get('lossless')}): {f['result']}{' **kept**' if f.get('keep') else ''}" for f in res["flags"]]
    par = res.get("parallel")
    if par and par.get("rows"):  # 2026-11-12 stage 6p: a report, the launch line above is for one request
        L += ["", "## Parallel requests", "", f"One request decodes at {par['plain_tps']} tok/s. Best: **{par['best_n']} at once**"
              + ("" if par["best_n"] > 1 else " (more requests gave no more total speed)") + ". Not in the launch line: the context per request "
              "is your call. To use N: `--max-running-requests N --cuda-graph-max-bs N` and FT_GRAPH_MAX_BS=N.", "",
              "| requests | total tok/s | per request tok/s | one request tok/s | KV tokens | max VRAM GiB | min RAM GiB |", "|---|---|---|---|---|---|---|"]
        L += [f"| {x['n']} | {x['total_tps'] if x['ok'] else 'FAIL: ' + str(x['error'])[:80]} | {' '.join(str(v) for v in x['per_req_tps'] or [])} | {x['single_tps'] or ''} "
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
           "notes": res["notes"], "ok": best is not None}
    if best is None:
        return out
    win = spec if spec and better(spec, best) else best
    keep_env = {f["name"]: f["value"] for f in res["flags"] if f.get("keep") and f["kind"] == "env"}
    keep_args = [x for f in res["flags"] if f.get("keep") and f["kind"] == "arg" for x in f["value"].split()]
    vis_ok = bool(confirm and confirm.get("ok") and confirm.get("vision"))
    fused = "fused" in win["args"]
    a, plain, spec_args, head = list(win["args"]), [], [], None
    i, picked = 0, 5000
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
    out.update(args=" ".join(args), cache=cache, head=head, env=keep_env, pick="spec" if win is spec else "plain",
               tps=tps(win), plain_tps=tps(best), spec_tps=tps(spec) if spec else None,
               p5k_tps=confirm.get("tps", {}).get("p5k") if confirm and confirm.get("ok") else None,
               confirm_ok=bool(confirm and confirm.get("ok")), vision=bool(inv["mmproj"]), vision_ok=vis_ok,
               min_ram_gib=min(ram) if ram else None, max_vram_gib=max(vram) if vram else None, load_s=win.get("load_s"),
               text_same=bool(win is best or spec.get("text") == best.get("text")))
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
        if a.plan_only or inv["verdict"] == "no-fit":
            if not a.plan_only:
                print("  stopping: the size check says it cannot fit")
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
    cp = os.path.join(PROBE_OUT, "cards", f"{a.tag or slug(m)}.md")
    with open(cp, "w", encoding="utf-8") as f:
        f.write(card(res))
    print(f"  card: {cp}")
    if a.result:
        keys = ("stage", "label", "args", "tests", "vision", "env", "ok", "tps", "error", "min_ram_gib", "max_vram_gib", "load_s", "when")
        w = dict(winner(res), card=cp, log=[{k: r.get(k) for k in keys} for r in res["log"]])
        with open(a.result, "w", encoding="utf-8") as f:
            json.dump(w, f, indent=1)
        print(f"  result: {a.result}")


if __name__ == "__main__":
    main()
