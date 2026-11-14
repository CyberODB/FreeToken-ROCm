"""Offline check of ft-fit's ladder: Runner.run is replaced by made-up rows, nothing loads, no GPU.
usage: py speedprobe\\slotcheck.py   (the last line counts the failed checks)
2026-11-13: rewritten for the flow rework (spec before slots, context floor, tune run, as-stamped final run, token map,
8-bit KV row). The older version is slotcheck.py.bak-20261113-flow. Tune files go to a temp folder, never speedprobe/tune."""
import importlib.util, json, os, tempfile, types

sp = importlib.util.spec_from_file_location("fit", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "ft-fit.py"))
fit = importlib.util.module_from_spec(sp); sp.loader.exec_module(fit)
TMP = tempfile.mkdtemp(prefix="slotcheck-")
fit.TUNE_DIR = os.path.join(TMP, "tune")
INV = {"model": r"X:\m.gguf", "experts": 256, "context_length": 262144, "verdict": "hybrid", "builtin_mtp": False, "sidecars": [],
       "mmproj": None, "ram_total_gib": 32, "arch": "a", "layers": 40, "experts_used": 8, "file_gib": 20, "experts_gib": 17,
       "dense_gib": 3, "expert_types": "IQ4_NL", "gate_up_q8_layers": [], "kv_bytes_per_token": 20000, "kv_gib_32k": 0.6,
       "mmproj_gib": None, "sidecar_exact": False, "host_need_text_gib": 20, "host_need_vision_gib": 20}
KV = {5000: 40000, 4000: 90000, 3000: 140000, 2000: 190000, 6500: 20000, 8000: 10000}
BAD = []


def check(name, ok, detail=""):
    print(f"{name}: {'OK' if ok else 'WRONG'} {detail}")
    if not ok: BAD.append(name)


def slots_of(args): return int(args[args.index("--moe-cache-size") + 1]) if "--moe-cache-size" in args else 5000


def go(name, edit=None, flags=(), inv=None, kv=None, **opts):
    """One ladder on made-up rows. A row is good by default: 88 tok/s, 800 tok/s prefill, KV by slot count (table kv or KV).
    edit(r, args, tests, env) changes it in place. Returns (res, winner, log)."""
    table = {**KV, **(kv or {})}

    def run(self, stage, label, args, tests="decode", vision=False, env=None, base_args=True, ctx=None):
        ts = tests.split(",")
        r = {"key": json.dumps([label, list(args), tests, vision, base_args, ctx]), "stage": stage, "label": label, "args": list(args), "tests": tests, "vision": vision, "env": {**self.base_env, **(env or {})}, "ok": True,
             "tps": {t: 88 for t in ts}, "prefill": {t: 800 for t in ts}, "kv_tokens": table.get(slots_of(args), 90000), "text": "same", "min_ram_gib": 5,
             "max_vram_gib": 15, "slots_auto": None, "kv_short": False, "base_args": base_args, "ctx": ctx}
        if edit: edit(r, args, tests, r["env"])
        if not r["ok"]: r["error"] = r.get("error") or "boom"
        self.runs += 1; self.log.append(r); return r
    fit.Runner.run = run
    fit.load_flags = lambda: [dict(f) for f in flags]
    a = types.SimpleNamespace(**{**dict(max_runs=10, load_timeout=600, tag="slotcheck-" + name, ctx=65536, fresh=True, no_slots=False, reserve_gib=""), **opts})
    res = fit.ladder(dict(inv or INV), a); res.update(tag=a.tag, ctx=res.pop("ctx_used"))
    w = fit.winner(res)
    for n in res["notes"]: print(f"   [{name}] note:", n[:230])
    return res, w, res["log"]


def stages(log, s): return [r for r in log if r["stage"] == s]


def clear_tune():
    for f in os.listdir(fit.TUNE_DIR) if os.path.isdir(fit.TUNE_DIR) else []: os.remove(os.path.join(fit.TUNE_DIR, f))


MTP = dict(INV, builtin_mtp=True)
spec_on = lambda args: "--speculative-algorithm" in args

# ---------- slots against the 64K floor ----------
res, w, log = go("floor-down")
fin = stages(log, "7-final")
check("floor-down picks the first count that holds 64K", (w["slots"], w["cache"]) == (4000, 4000) and "--moe-cache-size" not in w["args"], f"slots {w['slots']}")
check("floor-down: 3000 and 2000 never run", {slots_of(r["args"]) for r in stages(log, "6-slots")} == {5000, 4000})
check("floor-down: final run carries the pick, 5K prompt, model-max context", len(fin) == 1 and slots_of(fin[0]["args"]) == 4000 and fin[0]["tests"] == "decode,p5k" and fin[0]["ctx"] == 262144)
check("floor-down: floor met, context from the final run's KV", w["goal_ok"] is True and w["ctx"] == 90000 // 1024 * 1024 and w["ctx_goal"] == 65536 and w["ctx_floor"] == 32768 and w["ctx_verdict"] == "met", f"ctx {w['ctx']}")
check("floor-down: no second load for the context stage", not [r for r in stages(log, "8-ctx") if r["tests"] == "decode"])
card1 = fit.card(res)

res, w, log = go("floor-up", lambda r, a, t, e: r["tps"].update(decode={5000: 88, 6500: 95, 8000: 96}.get(slots_of(a), 88)), kv={5000: 100000, 6500: 80000, 8000: 70000})
check("floor-up: more slots while the floor holds and decode gains", w["slots"] == 6500 and w["cache"] == 6500, f"slots {w['slots']}")
res, w, log = go("floor-up-kv", lambda r, a, t, e: r["tps"].update(decode=99 if slots_of(a) == 6500 else 88), kv={5000: 100000, 6500: 50000})
check("floor-up: a faster count under the floor is refused", w["slots"] == 5000, f"slots {w['slots']}")
res, w, log = go("floor-never", kv={5000: 40000, 4000: 45000, 3000: 50000, 2000: 55000})
# 2026-11-14: 55K is under the 64K goal but over the 32K floor: usable, "under goal"
check("floor-never: most context kept, under goal (not failed)", w["slots"] == 2000 and w["goal_ok"] is False and w["floor_ok"] is True
      and w["ctx_verdict"] == "under goal" and "**under goal**" in fit.card(res) and any("no slot count holds" in n for n in res["notes"]))
res, w, log = go("floor-fail", kv={5000: 20000, 4000: 22000, 3000: 25000, 2000: 28000})
check("under the 32K floor: failed, 8-bit KV rescue tried", w["floor_ok"] is False and w["ctx_verdict"] == "failed" and "Failed: context floor" in fit.card(res)
      and bool(stages(log, "8-kv8")))
res, w, log = go("goal-arg", ctx_goal=120000, ctx_floor=90000)
check("--ctx-goal 120000 --ctx-floor 90000: 3000 slots, both reported", w["slots"] == 3000 and w["ctx_goal"] == 120000 and w["ctx_floor"] == 90000)
res, w, log = go("4000-fails", lambda r, a, t, e: r.update(ok=slots_of(a) != 4000))
check("4000 fails: 5000 kept", w["slots"] == 5000 and w["cache"] == 5000)
res, w, log = go("floor-arg", ctx_floor=120000)
check("--ctx-floor 120000: 3000 slots", w["slots"] == 3000 and w["ctx_floor"] == 120000, f"slots {w['slots']}")

# ---------- spec before slots; token map ----------
mtp = lambda r, a, t, e: r["tps"].update(decode=120) if spec_on(a) else None
res, w, log = go("spec-first", mtp, inv=MTP)
order = [r["stage"] for r in log]
check("spec runs before the slot sweep", order.index("5-spec") < order.index("6-slots"))
check("slots are swept on the spec config", all(spec_on(r["args"]) for r in stages(log, "6-slots")) and w["pick"] == "spec" and w["head"]["kind"] == "mtp" and w["slots"] == 4000)
check("final run is the spec config with its slots", spec_on(stages(log, "7-final")[0]["args"]) and w["tps"] == 120 and w["final_tps"] == 120)
res, w, log = go("spec-slower", lambda r, a, t, e: r["tps"].update(decode=60) if spec_on(a) else None, inv=MTP)
check("a slower spec is dropped before the slots", not any(spec_on(r["args"]) for r in stages(log, "6-slots") + stages(log, "7-final")) and w["pick"] == "plain")
tm = os.path.join(TMP, "map.bin"); open(tm, "wb").write(b"x")
tmap = lambda gain: (lambda r, a, t, e: r["tps"].update(decode=gain if "--speculative-token-map" in a else 120) if spec_on(a) else None)
res, w, log = go("tmap-kept", tmap(130), inv=MTP, token_map=tm)
check("token map kept when faster", "--speculative-token-map" in w["args"] and w["tps"] == 130 and w["head"]["kind"] == "mtp", w["args"])
res, w, log = go("tmap-same", tmap(121), inv=MTP, token_map=tm)
check("token map dropped inside the noise band", "--speculative-token-map" not in w["args"])
res, w, log = go("tmap-missing", mtp, inv=MTP, token_map=os.path.join(TMP, "nope.bin"))
check("a missing token map file is a note, not a run", not any("--speculative-token-map" in r["args"] for r in log), str([n for n in res["notes"] if "map" in n])[:200])

# ---------- tune run, then a fit that uses it ----------
FLAGS = [{"id": "good", "kind": "env", "name": "FT_GOOD", "value": "1", "lossless": True, "status": "safe"},
         {"id": "once", "kind": "env", "name": "FT_ONCE", "value": "1", "lossless": True, "status": "safe"},
         {"id": "lossy", "kind": "arg", "name": "--lossy", "value": "--lossy 1", "lossless": False, "status": "safe"}]
clear_tune()
res, w, log = go("no-tune")
check("no tune file: a note, no flag or parallel runs", any("no tune result" in n for n in res["notes"]) and not stages(log, "6-flag") and not stages(log, "6p-parallel") and w["flags"] == [])


def tune_edit(r, a, t, e):
    if e.get("FT_GOOD"): r["tps"]["decode"] = 95
    if e.get("FT_ONCE"): r["tps"]["decode"] = 95 if not r["label"].endswith("-again") else 88
    if "--lossy" in a: r["tps"]["decode"] = 99


res, w, log = go("tune", tune_edit, flags=FLAGS, tune=True)
print("   tune stages:", sorted({r["stage"] for r in log}), "labels:", [r["label"] for r in log if r["label"].endswith("-again")])
tf = json.load(open(os.path.join(fit.TUNE_DIR, "a-moe.json"), encoding="utf-8"))
keep = {f["id"]: bool(f.get("keep")) for f in tf["flags"]}
check("tune: a gain seen twice is kept; once, or lossy, is not", keep == {"good": True, "once": False, "lossy": False}, str(keep))
check("tune: winners run twice", len([r for r in log if r["label"].endswith("-again")]) >= 2)
check("tune: the parallel report is saved", bool((tf.get("parallel") or {}).get("rows")), str(tf.get("parallel"))[:160])
check("tune: result is marked, has no config", w.get("tune") is True and "args" not in w and not stages(log, "7-final") and not stages(log, "5-spec"))
card_t = fit.card(res)
res, w, log = go("fit-after-tune", lambda r, a, t, e: r["tps"].update(decode=95) if e.get("FT_GOOD") else None)
fin = stages(log, "7-final")
check("fit after tune: kept flag is in the final run, and on the card", fin[0]["env"] == {"FT_GOOD": "1"} and w["env"] == {"FT_GOOD": "1"} and w["flags"] == ["good"], f"{fin[0]['env']} {w['env']} {w['flags']}")
check("fit after tune: no flag before the final run", not any(r["env"] for r in log if r["stage"] not in ("7-final", "8-ctx", "8-kv8")))
check("fit after tune: speed is the final run's", w["tps"] == 95 and w["ladder_tps"] == 88, f"{w['tps']} {w.get('ladder_tps')}")
check("fit after tune: fill test and parallel report use the tune", all(r["env"] == {"FT_GOOD": "1"} for r in stages(log, "8-ctx")) and bool((w.get("parallel") or {}).get("rows"))
      and (w.get("tune_from") or {}).get("class") == "a-moe", str(w.get("tune_from"))[:160])
res, w, log = go("flags-hurt", lambda r, a, t, e: r["tps"].update(decode=70) if e.get("FT_GOOD") else None)
check("tune flags that slow this model are dropped after one run without them", w["env"] == {} and w["flags"] == [] and w["tps"] == 88 and len(stages(log, "7-final")) == 2, f"{w['env']} {w['tps']} {len(stages(log, '7-final'))}")
res, w, log = go("flags-break", lambda r, a, t, e: r.update(ok=not e.get("FT_GOOD")))
check("tune flags that break the load are dropped", w["env"] == {} and w["confirm_ok"] and w["tps"] == 88, f"{w['env']} {w['confirm_ok']} {[r['label'] for r in stages(log, '7-final')]}")
clear_tune()

# ---------- final run: failures ----------
res, w, log = go("final-kvshort", lambda r, a, t, e: r.update(ok=False, kv_short=True) if r["stage"] == "7-final" and r["ctx"] == 262144 else None)
fin = stages(log, "7-final")
check("final run: a KV shortfall lowers the context cap", len(fin) == 2 and fin[1]["ctx"] == 196608 and w["confirm_ok"], str([r["ctx"] for r in fin]))
res, w, log = go("final-ratio", lambda r, a, t, e: r.update(ok=False) if r["stage"] == "7-final" and "0.80" in a else None)
check("final run: another failure lowers the memory ratio, and the card follows", w["confirm_ok"] and "--memory-ratio 0.75" in w["args"], w["args"])
res, w, log = go("final-dead", lambda r, a, t, e: r.update(ok=False) if r["stage"] == "7-final" else None)
check("final run never passes: said so, ladder speed shown", w["confirm_ok"] is False and w["final_tps"] is None and w["tps"] == 88, f"{w['confirm_ok']} {w.get('final_tps')} {w['tps']}")

# ---------- 8-bit KV row ----------
kv8 = lambda r, a, t, e: r.update(kv_tokens=170000, text="other", tps={"decode": 85}) if "--kv-cache-dtype" in a else None
os.environ["FT_FIT_KV8"] = "1"  # 2026-11-14: 8-bit KV runs on its own only as a floor rescue; these two force it
res, w, log = go("kv8", kv8)
k = w["kv8"] or {}
check("8-bit KV: reported, never in the launch line", bool(k.get("ok")) and k.get("kv_tokens") == 170000 and "--kv-cache-dtype" not in w["args"] and "8-bit KV" in fit.card(res), str(k)[:220])
res, w, log = go("kv8-fails", lambda r, a, t, e: r.update(ok=False, error="needs bfloat16") if "--kv-cache-dtype" in a else None)
check("8-bit KV: a failed load is a row with its error, the card stays good", (w["kv8"] or {}).get("ok") is False and w["confirm_ok"], str(w["kv8"])[:200])
os.environ.pop("FT_FIT_KV8", None)
res, w, log = go("kv8-off", no_kv8=True)
check("--no-kv8 skips the row", w["kv8"] is None and not stages(log, "8-kv8"))

# ---------- older stages still work ----------
res, w, log = go("auto-slots", lambda r, a, t, e: r.update(slots_auto=3500) if "--moe-cache-auto" in a else None, slots_mode="auto")
# 2026-11-13: the card keeps --moe-cache-auto + the reserve, not the count it resolved to (free VRAM moves between loads)
check("auto slots: the engine's count is reported, the card keeps --moe-cache-auto", w["slots"] == 3500 and "--moe-cache-auto" in w["args"], f"slots {w.get('slots')}")
head = os.path.join(TMP, "dflash.bin"); open(head, "wb").write(b"x")
res, w, log = go("dflash", lambda r, a, t, e: r["tps"].update(decode=140 if "dflash" in a else 120) if spec_on(a) else None, inv=MTP, dflash_head=head)
check("DFlash faster than MTP: kept, swept and stamped", w["head"]["kind"] == "dflash" and all("dflash" in r["args"] for r in stages(log, "6-slots") + stages(log, "7-final")), str(w["head"]))
res, w, log = go("dense", inv=dict(INV, experts=None))
check("dense model: no slot sweep, final run still happens", not stages(log, "6-slots") and w["confirm_ok"] and w.get("slots") is None, f"{w.get('slots')} {w.get('cache')}")
res, w, log = go("no-load", lambda r, a, t, e: r.update(ok=False))
check("nothing loads: no config, no crash", w["ok"] is False and "No config loaded" in fit.card(res))
res, w, log = go("reserve", reserve_gib="2")
check("desktop reserve is on every run and in the card's env", all(r["env"].get("FREETOKEN_DESKTOP_RESERVE_GB") == "2" for r in log) and w["env"] == {"FREETOKEN_DESKTOP_RESERVE_GB": "2"}, str(w["env"]))

print("\n" + card1[:1700])
print("\n" + card_t[:500])
print(f"\n{len(BAD)} of the 2026-11-13 checks failed" + (": " + "; ".join(BAD) if BAD else ""))
