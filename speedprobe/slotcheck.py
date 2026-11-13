"""Offline check of ft-fit's slot sweep (stage 4s): Runner.run is replaced by made-up rows, nothing loads.
usage: py speedprobe\\slotcheck.py   (prints the pick for three made-up cases and one card)
2026-11-12: also the generalized flag stage, the parallel stage, --slots-mode auto and the DFlash head (undo: slotcheck.py.bak-20261112-stages).
The made-up rows set every field themselves (slots_auto, per_req): no log file is read. The last line counts the failed checks."""
import importlib.util, json, types

sp = importlib.util.spec_from_file_location("fit", r"E:\freetoken-021\scripts\ft-fit.py")
fit = importlib.util.module_from_spec(sp); sp.loader.exec_module(fit)
INV = {"model": r"X:\m.gguf", "experts": 256, "context_length": 262144, "verdict": "hybrid", "builtin_mtp": False, "sidecars": [],
       "mmproj": None, "ram_total_gib": 32, "arch": "a", "layers": 40, "experts_used": 8, "file_gib": 20, "experts_gib": 17,
       "dense_gib": 3, "expert_types": "IQ4_NL", "gate_up_q8_layers": [], "kv_bytes_per_token": 20000, "kv_gib_32k": 0.6,
       "mmproj_gib": None, "sidecar_exact": False, "host_need_text_gib": 20, "host_need_vision_gib": 20}


def case(name, table, ask):
    """table: slots -> (decode, prefill, kv)"""
    def run(self, stage, label, args, tests="decode", vision=False, env=None, base_args=True, ctx=None):
        n = int(args[args.index("--moe-cache-size") + 1]) if "--moe-cache-size" in args else 5000
        d, p, kv = table[n]
        r = {"stage": stage, "label": label, "args": args, "tests": tests, "vision": vision, "env": env or {}, "ok": d > 0,
             "tps": {t: d for t in tests.split(",")}, "prefill": {"p5k": p}, "kv_tokens": kv, "text": "same", "min_ram_gib": 5,
             "max_vram_gib": 15, "error": "boom"}
        self.log.append(r); return r
    fit.Runner.run = run
    fit.load_flags = lambda: []
    a = types.SimpleNamespace(max_runs=10, load_timeout=600, tag="slotcheck-" + name, ctx=65536, fresh=True, no_slots=False, reserve_gib="")
    res = fit.ladder(INV, a); res.update(tag=a.tag, ctx=res.pop("ctx_used"))
    w = fit.winner(res)
    got = (w["slots"], w["cache"], "--moe-cache-size" in w["args"])
    print(f"{name}: slots {w['slots']} cache {w['cache']} args [{w['args']}] {'OK' if got == (ask, ask, False) else 'WRONG, wanted ' + str(ask)}")
    for n in res["notes"]: print("   note:", n)
    return res


case("all-fine", {5000: (88, 800, 40000), 4000: (88, 790, 90000), 3000: (87, 780, 140000), 2000: (86, 760, 190000)}, 2000)
case("3000-slow", {5000: (88, 800, 40000), 4000: (87, 790, 90000), 3000: (70, 500, 140000), 2000: (60, 300, 190000)}, 4000)
case("no-kv-gain", {5000: (88, 800, 40000), 4000: (88, 800, 40000), 3000: (88, 800, 40000), 2000: (88, 800, 40000)}, 5000)
r = case("4000-fails", {5000: (88, 800, 40000), 4000: (0, 0, None), 3000: (88, 800, 1), 2000: (88, 800, 1)}, 5000)
print(fit.card(case("all-fine", {5000: (88, 800, 40000), 4000: (88, 790, 90000), 3000: (87, 780, 140000), 2000: (86, 760, 190000)}, 2000))[:1500])


# ---------- 2026-11-12: flags with values/metric, parallel stage, --slots-mode auto, DFlash ----------
BAD = []


def check(name, ok, detail=""):
    print(f"{name}: {'OK' if ok else 'WRONG'} {detail}")
    if not ok: BAD.append(name)


def ladder2(name, edit, flags=(), **opts):
    """edit(r, args, tests, env, base_args) changes a default good row (88 tok/s decode, 800 tok/s prefill, 40000 KV tokens) in place."""
    def run(self, stage, label, args, tests="decode", vision=False, env=None, base_args=True, ctx=None):
        ts = tests.split(",")
        r = {"stage": stage, "label": label, "args": args, "tests": tests, "vision": vision, "env": env or {}, "ok": True,
             "tps": {t: 88 for t in ts}, "prefill": {t: 800 for t in ts}, "kv_tokens": 40000, "text": "same", "min_ram_gib": 5,
             "max_vram_gib": 15, "slots_auto": None, "base_args": base_args}
        edit(r, args, tests, env or {}, base_args)
        if not r["ok"]: r["error"] = "boom"
        self.log.append(r); return r
    fit.Runner.run = run
    fit.load_flags = lambda: [dict(f) for f in flags]
    a = types.SimpleNamespace(**{**dict(max_runs=10, load_timeout=600, tag="slotcheck-" + name, ctx=65536, fresh=True, no_slots=True, reserve_gib="",
                                        slots_mode="sweep", kv_reserve=0, dflash_head=None, no_parallel=True), **opts})
    res = fit.ladder(INV, a); res.update(tag=a.tag, ctx=res.pop("ctx_used"))
    return res, fit.winner(res), fit.card(res)


FETCH = {"id": "max-fetch", "kind": "arg", "name": "--moe-hybrid-max-fetch", "values": ["--moe-hybrid-max-fetch 2", "--moe-hybrid-max-fetch 4"],
         "tests": "decode,p5k", "metric": "prefill:p5k", "lossless": True, "status": "safe"}
arg = lambda args, k: args[args.index(k) + 1] if k in args else None
early = lambda r: "confirm" not in r["label"] and "ctx" not in r["stage"]  # the made-up effect stays out of stages 7 and 8


def fetch(decode4):  # fetch 4: 5K prefill +10%, decode as given; fetch 2: no change
    def edit(r, args, tests, env, base_args):
        if arg(args, "--moe-hybrid-max-fetch") == "4" and early(r):
            r["prefill"]["p5k"] = 880; r["tps"]["decode"] = decode4
    return edit


res, w, c = ladder2("flag-prefill-kept", fetch(88), [FETCH])
f = res["flags"][0]
check("flag prefill +10%, decode equal", f["keep"] and f["value"] == "--moe-hybrid-max-fetch 4" and w["args"].endswith("--moe-hybrid-max-fetch 4") and "**kept**" in c,
      f"keep {f['keep']} args [{w['args']}]")
for v in f["variants"]: print("   variant:", {k: v[k] for k in ("label", "value", "metric", "ref", "text_same", "pass")})
print("   ref run:", [r["label"] for r in res["log"] if "-ref-" in r["label"]], "| result:", f["result"][:230])
res, w, c = ladder2("flag-decode-drops", fetch(79.2), [FETCH])
f = res["flags"][0]
check("flag prefill +10%, decode -10%", not f["keep"] and "max-fetch" not in w["args"], f"keep {f['keep']} worse {f['variants'][1]['worse']} args [{w['args']}]")
res, w, c = ladder2("flag-unknown", fetch(88), [dict(FETCH, lossless="unknown")])
check("flag lossless unknown is only reported", not res["flags"][0]["keep"] and "reported only" in res["flags"][0]["result"])
THREADS = [{"id": f"cpu-threads-{n}", "kind": "arg", "name": "--moe-cpu-threads", "value": f"--moe-cpu-threads {n}", "lossless": True, "status": "safe"} for n in (6, 8)]


def threads(r, args, tests, env, base_args):
    if arg(args, "--moe-cpu-threads") and early(r): r["tps"]["decode"] = {"6": 92, "8": 95}[arg(args, "--moe-cpu-threads")]


res, w, c = ladder2("flag-same-name", threads, THREADS)
check("two kept flags of one name: the better stays", [f["keep"] for f in res["flags"]] == [False, True] and w["args"].count("--moe-cpu-threads") == 1
      and w["args"].endswith("threads 8"), f"args [{w['args']}]")


def par(c2, c4, ok4=True):
    def edit(r, args, tests, env, base_args):
        n = arg(args, "--max-running-requests")
        if n:
            assert arg(args, "--cuda-graph-max-bs") == n and env.get("FT_GRAPH_MAX_BS") == n and tests == f"decode,c{n}", (args, env, tests)
            r["tps"][f"c{n}"] = {"2": c2, "4": c4}[n]; r["per_req"] = {f"c{n}": [r["tps"][f"c{n}"] / int(n)] * int(n)}
            r["ok"] = n != "4" or ok4
    return edit


for name, e, want in (("parallel 2:150 4:200", par(150, 200), 4), ("parallel 2:150 4:fails", par(150, 200, False), 2), ("parallel no gain", par(89, 88), 1)):
    res, w, c = ladder2("par", e, no_parallel=False)
    p = w["parallel"]
    check(name, p["best_n"] == want and "max-running" not in w["args"] and "cuda-graph" not in w["args"] and "## Parallel requests" in c,
          f"best_n {p['best_n']} rows {[(x['n'], x['ok'], x['total_tps']) for x in p['rows']]} args [{w['args']}]")
print(c[c.index("## Parallel requests"):c.index("## Notes") if "## Notes" in c else None])
res, w, c = ladder2("par-off", par(150, 200), no_parallel=True)
check("--no-parallel", "parallel" not in w and not any(r["stage"] == "6p-parallel" for r in res["log"]))


def auto(slots, seen):
    def edit(r, args, tests, env, base_args):
        if "--moe-cache-auto" in args:
            seen.append((list(args), base_args, tests)); r["slots_auto"] = slots; r["kv_tokens"] = 120000
    return edit


seen = []
res, w, c = ladder2("slots-auto", auto(3600, seen), no_slots=False, slots_mode="auto")
a0, base0, t0 = seen[0]
check("slots auto: the run", len(seen) == 1 and base0 is False and "--moe-cache-size" not in a0 and a0[:2] == ["--moe-cpu-layers", "auto"]
      and arg(a0, "--kv-reserve-tokens") == "65536" and t0 == "decode,p5k", f"[{' '.join(a0)}] base_args {base0}")
check("slots auto: card cache", w["cache"] == 3600 and w["slots"] == 3600 and "moe-cache" not in w["args"] and "--moe-cache-size 3600" in c and "**3600 slots**" in c,
      f"cache {w['cache']} args [{w['args']}]")
seen = []
res, w, c = ladder2("slots-auto-unparsed", auto(None, seen), no_slots=False, slots_mode="auto", kv_reserve=32768)
check("slots auto: count not in the log", w["cache"] == 5000 and arg(seen[0][0], "--kv-reserve-tokens") == "32768" and any("does not name" in n for n in res["notes"]),
      f"cache {w['cache']}")


def dfl(r, args, tests, env, base_args):
    if arg(args, "--speculative-algorithm") == "dflash" and early(r): r["tps"]["decode"] = {"2": 100, "3": 120, "4": 110}[arg(args, "--speculative-num-steps")]


res, w, c = ladder2("dflash", dfl, dflash_head=__file__)
check("dflash head", w["pick"] == "spec" and w["head"] == {"kind": "dflash", "path": __file__, "steps": 3} and w["tps"] == 120, f"head kind {w['head']['kind']} k {w['head']['steps']} tps {w['tps']}")
print(f"{len(BAD)} of the 2026-11-12 checks failed" + (": " + ", ".join(BAD) if BAD else ""))
