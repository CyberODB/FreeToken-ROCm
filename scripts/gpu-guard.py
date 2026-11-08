"""GPU guard for FreeToken tests on Windows: VRAM floor, kill switch, crash journal.

The desktop launcher runs these two steps for every test:
  gpu-guard.py --check --name NAME     exit 3 if NAME is a suspected PC-killer (launcher stops)
  gpu-guard.py --watch --name NAME     background watcher (hidden), runs while FreeToken runs
Other uses:
  gpu-guard.py --report                journal summary and the suspected-killer list
  gpu-guard.py --clear NAME            forget a suspected killer
  gpu-guard.py --name N [opts] -- CMD  run one command under the same rules (kills CMD's tree)

Rules:
  * Kill when any floor is crossed for --samples samples in a row (1 s apart): free dedicated VRAM
    under --min-free-gib (0.4; hybrid/offload fill VRAM on purpose, ratio 0.85 sits at ~15.1 of 16 GiB),
    available host RAM under --min-ram-gib (1.5) or free commit under --min-commit-gib (2.0).
    --watch kills the FreeToken GPU processes; wrap mode kills CMD's tree plus any python whose
    command line has --stale (default "--port 1925", an orphaned probe server). Browsers etc. are never touched.
  * Wrap mode also kills such leftovers before the run and after it, waits until the killed PIDs are gone
    and VRAM is back near the pre-run level ("recovered in N s"), and exits 4 so a sweep moves on.
    CMD runs at normal priority (--low-prio for below-normal, which children inherit and which skews tok/s).
  * Every sample (VRAM, RAM, commit) is written to the journal and fsynced, so the last line before a
    freeze survives a hard reset.
  * A watcher session with no "end" line after a reboot is looked at: if Windows logged
    Kernel-Power event 41 (unclean shutdown) after the last sample, that test is a suspected
    killer and --check refuses it until --force or --clear. A clean restart is not blamed.
Every kill also prints a plain-language "WHY / WHAT TO TRY" line, journals it as `tip`, and writes last-stop.txt in $FT_GUARD_DIR.
Journal and killer list: $FT_GUARD_DIR (default %LOCALAPPDATA%\\freetoken-gpu-guard).
Profiles (--profile or env FT_GUARD_PROFILE, default "debug"; explicit flags/env below still override a profile):
  debug   strict, for GPU/expert/layer debugging where a crash is likely: free VRAM 0.2, RAM 1.5, commit 2.0 GiB, 2 samples
  normal  everyday model serving (Hermes chat): VRAM 0.15, RAM 0.4, commit 0.75 GiB, 4 samples in a row (fires less);
          plus a soft RAM floor of 1.0 GiB that only counts while Pages Input/sec >= 2000 (real paging, not just low RAM)
Env: FT_GUARD_MIN_FREE_GIB (0.4), FT_GUARD_MIN_RAM_GIB (1.5), FT_GUARD_MIN_COMMIT_GIB (2.0), FT_GUARD_SAMPLES (2),
     FT_GUARD_STALE ("--port 1925"), FT_GUARD_TOTAL_GIB (16), FT_GUARD_MATCH ("freetoken").
"""
import argparse, csv, io, json, os, subprocess, sys, threading, time

PROFILES = {  # floors (GiB) and samples in a row; "debug" = the original behaviour
    "debug":  dict(free=0.2,  ram=1.5, commit=2.0,  samples=2, soft=0.0, page_in=0),  # VRAM 0.2: this model sits at ~0.33 free by design; 0.4 killed it in a loop
    # RAM 0.4 (was 1.0): the 35B hybrid model idles at only 1.1-2.0 GiB available and a Hermes prefill dips to ~0.65, so 1.0
    # killed healthy runs 20+ times on 2026-11-08 (commit free was ~11 GiB and the sampler never stalled). Undo: ram=1.0, samples=3.
    # soft=1.0/page_in=2000: below 1.0 GiB available RAM only counts as a breach while the box is also paging in (>2000 pages/s,
    # i.e. real thrash); ram=0.4 stays as the hard floor with no paging test. Undo: soft=0.0.
    "normal": dict(free=0.15, ram=0.4, commit=0.75, samples=4, soft=1.0, page_in=2000),
}

DIR = os.environ.get("FT_GUARD_DIR") or os.path.join(
    os.environ.get("LOCALAPPDATA", "."), "freetoken-gpu-guard")
JOURNAL = os.path.join(DIR, "journal.jsonl")
KILLERS = os.path.join(DIR, "killers.json")
GIB = 1024 ** 3
NOWIN = 0x08000000  # CREATE_NO_WINDOW


def ps(cmd, timeout=60):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                       text=True, timeout=timeout, creationflags=NOWIN)
    return r.stdout.strip()


def boot_id():
    return ps("(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('s')")


def jwrite(**kw):
    kw["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(DIR, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")
        f.flush()
        os.fsync(f.fileno())


def jread():
    out = []
    if os.path.exists(JOURNAL):
        for line in open(JOURNAL, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def killers():
    try:
        return json.load(open(KILLERS))
    except Exception:
        return {}


def save_killers(k):
    os.makedirs(DIR, exist_ok=True)
    json.dump(k, open(KILLERS, "w"), indent=1)


def pid_alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True,
                         text=True, creationflags=NOWIN).stdout
    return str(pid) in out


def resolve_unclean():
    """Close any session that has no end line. Blame it only if Windows logged an unclean shutdown."""
    cur = last = None
    for r in jread():
        if r["ev"] == "start":
            cur, last = r, None
        elif r["ev"] == "vram" and cur:
            last = r
        elif r["ev"] == "end":
            cur = last = None
    if not cur:
        return
    if cur.get("boot") == boot_id():
        if cur.get("pid") and pid_alive(cur["pid"]):
            return  # still running
        jwrite(ev="end", name=cur["name"], how="watcher gone, same boot (not blamed)")
        return
    since = (last or cur)["t"]
    hit = ps(f"Get-WinEvent -FilterHashtable @{{LogName='System';Id=41;StartTime='{since}'}} "
             f"-MaxEvents 1 -ErrorAction SilentlyContinue | % {{ $_.TimeCreated.ToString('s') }}")
    if hit:
        k = killers()
        k[cur["name"]] = {"started": cur["t"], "last_sample": since,
                          "last_vram_used_gib": last and last["used_gib"],
                          "unclean_shutdown_logged": hit}
        save_killers(k)
        jwrite(ev="end", name=cur["name"], how="UNCLEAN: PC died (Kernel-Power 41 at " + hit + ")")
        print(f"[gpu-guard] '{cur['name']}' was running when the PC died ({hit}). Last VRAM used "
              f"{last and last['used_gib']} GiB at {since}. Marked as suspected killer.", flush=True)
    else:
        jwrite(ev="end", name=cur["name"], how="clean restart (not blamed)")


def sample_stream(si=1):
    """Yield (used dedicated VRAM bytes of the busiest adapter, available RAM GiB, free commit GiB, pages read in/s) every si s."""
    p = subprocess.Popen(["typeperf", r"\GPU Adapter Memory(*)\Dedicated Usage", r"\Memory\Available MBytes",
                          r"\Memory\Committed Bytes", r"\Memory\Commit Limit", r"\Memory\Pages Input/sec", "-si", str(si)],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         creationflags=NOWIN)
    hdr = None
    try:
        for line in p.stdout:
            line = line.strip()
            if not line.startswith('"'):
                continue
            row = next(csv.reader([line]))
            if hdr is None:
                hdr = row
                gi = [i for i, h in enumerate(hdr) if "GPU Adapter Memory" in h]
                ai, ci, li, pi = (next(i for i, h in enumerate(hdr) if h.endswith(s))
                                  for s in ("Available MBytes", "Committed Bytes", "Commit Limit", "Pages Input/sec"))
                continue
            try:
                yield (max(float(row[i]) for i in gi if row[i].strip()), float(row[ai]) / 1024,
                       (float(row[li]) - float(row[ci])) / GIB, float(row[pi]))
            except (ValueError, IndexError):
                continue
    finally:
        p.kill()


class Limits:
    """Breach rule shared by --watch and wrap mode: any floor crossed for `samples` samples in a row.

    Hybrid/offload fill VRAM on purpose (ratio 0.85 sits at ~15.1 of 16 GiB), so the VRAM floor is low;
    host RAM and commit are the floors that matter (pinned banks + an orphaned server took RAM to 2.8 GiB)."""
    def __init__(self, a):
        self.a, self.low = a, 0

    def check(self, used, avail, cfree, pin=0.0):
        a, free = self.a, self.a.total_gib - used / GIB
        soft = a.soft_ram_gib and avail < a.soft_ram_gib and pin >= a.page_in  # low RAM AND thrashing, not just low
        why = (f"free VRAM {free:.2f} GiB under {a.min_free_gib}" if free < a.min_free_gib else
               f"available RAM {avail:.2f} GiB under {a.min_ram_gib}" if avail < a.min_ram_gib else
               f"available RAM {avail:.2f} GiB under {a.soft_ram_gib} while paging in {pin:.0f} pages/s" if soft else
               f"free commit {cfree:.2f} GiB under {a.min_commit_gib}" if cfree < a.min_commit_gib else None)
        self.low = self.low + 1 if why else 0
        return why if self.low >= a.samples else None


def proc_list(match):
    js = ps("Get-CimInstance Win32_Process | ? { $_.CommandLine } | "
            "select ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress")
    procs = json.loads(js or "[]")
    procs = procs if isinstance(procs, list) else [procs]
    me = os.getpid()
    return [p for p in procs if match.lower() in p["CommandLine"].lower() and p["ProcessId"] != me
            and "gpu-guard" not in p["CommandLine"].lower()]


def kill_pids(pids):
    for pid in pids:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, creationflags=NOWIN)


def wait_recovered(pids, base_used, timeout=45):
    """After a kill: wait until the PIDs are gone and VRAM is back near the pre-run level. Returns seconds or None."""
    t0 = time.time()
    for used, _, _, _ in sample_stream():
        if not any(pid_alive(p) for p in pids) and used / GIB < base_used + 0.75:
            return round(time.time() - t0, 1)
        if time.time() - t0 > timeout:
            return None


def ft_gpu_pids(match):
    """PIDs matching the command-line filter that hold dedicated VRAM right now."""
    out = subprocess.run(["typeperf", r"\GPU Process Memory(*)\Dedicated Usage", "-sc", "1"],
                         capture_output=True, text=True, creationflags=NOWIN).stdout
    lines = [ln for ln in out.splitlines() if ln.startswith('"')][:2]  # typeperf prints blank lines first
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if len(rows) < 2:
        return []
    used = {}
    for h, v in zip(rows[0][1:], rows[1][1:]):
        try:
            pid = int(h.split("pid_")[1].split("_")[0])
            used[pid] = used.get(pid, 0) + float(v)
        except (IndexError, ValueError):
            pass
    me = os.getpid()
    js = ps("Get-CimInstance Win32_Process | ? { $_.CommandLine } | "
            "select ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress")
    procs = json.loads(js or "[]")
    procs = procs if isinstance(procs, list) else [procs]
    # 2026-11-07: ft serve's engine runs in a multiprocessing child whose command line has no "freetoken" in it
    # ("spawn_main(parent_pid=...)"), and that child holds the VRAM. Count every descendant of a matching process too.
    ok = {p["ProcessId"] for p in procs
          if match.lower() in p["CommandLine"].lower() and "gpu-guard" not in p["CommandLine"].lower()}
    for _ in range(6):
        ok |= {p["ProcessId"] for p in procs if p["ParentProcessId"] in ok}
    hits = []
    for p in procs:
        if p["ProcessId"] in ok and p["ProcessId"] != me and used.get(p["ProcessId"], 0) > 0:
            hits.append((p["ProcessId"], used[p["ProcessId"]]))
    return hits


def kill_hint(why):
    """Plain-language reason and the cheapest thing to try, from a Limits.check() reason string."""
    w = why or ""
    if w.startswith("free VRAM"):
        return ("The GPU ran out of free memory. Try, in order: lower the context (--max-seq-len-override) or KV (--num-pages, "
                "--kv-reserve-tokens) first; then a smaller --moe-cache-size; then --memory-ratio down 0.05; "
                "close apps that hold VRAM (browser hardware acceleration, games).")
    if "while paging in" in w:
        return ("The PC was out of RAM and swapping to disk, which is what freezes it. Try: close big apps (browser, Discord, "
                "Claude) before loading, lower the context or KV so less is resident, or serve a smaller quant.")
    if w.startswith("available RAM"):
        return ("Host RAM ran critically low. Try: close big apps, lower the context or KV, or lower the pin budget "
                "(FREETOKEN_PIN_BUDGET_GB) together with --moe-cpu-layers auto, or serve a smaller quant.")
    if w.startswith("free commit"):
        return "Windows ran nearly out of commit (RAM + page file). Try: close big apps or enlarge the page file."
    if w.startswith("timeout"):
        return "The run hit its time limit. Try: a shorter test, or check the serve log for a stuck load."
    return None


def note_stop(name, why, pids=None):
    """Journal-side human note: a hint line plus last-stop.txt (the file to read when a model vanished)."""
    tip = kill_hint(why)
    if tip:
        print(f"[gpu-guard] WHY / WHAT TO TRY: {tip}", flush=True)
    try:
        os.makedirs(DIR, exist_ok=True)
        with open(os.path.join(DIR, "last-stop.txt"), "w", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  guard stopped '{name}'\nWhy: {why}\n"
                    f"{'What to try: ' + tip if tip else ''}\n")
    except OSError:
        pass
    return tip


def kill_ft(match, name, why):
    hits = ft_gpu_pids(match)
    tip = kill_hint(why)
    jwrite(ev="kill", name=name, why=why, pids=[h[0] for h in hits], **({"tip": tip} if tip else {}))
    print(f"[gpu-guard] KILL '{name}': {why}; pids {[h[0] for h in hits]}", flush=True)
    note_stop(name, why)
    for pid, _ in hits:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                       creationflags=NOWIN)
    return bool(hits)


def jsample(name, a, lastw, used, avail, cfree, now, pin=0.0):
    """Journal a sample when it moved, every 10 s, or near a floor (fsynced, survives a hard reset)."""
    used_g, free = used / GIB, a.total_gib - used / GIB
    near = free < a.min_free_gib + 1 or avail < max(a.min_ram_gib, a.soft_ram_gib) + 2 or cfree < a.min_commit_gib + 2
    if abs(used_g - lastw[0]) > 0.05 or abs(avail - lastw[2]) > 0.5 or now - lastw[1] >= 10 or near:
        jwrite(ev="vram", name=name, used_gib=round(used_g, 2), free_gib=round(free, 2),
               ram_gib=round(avail, 2), commit_gib=round(cfree, 2), pagein=round(pin))
        return (used_g, now, avail)
    return lastw


def override_profile():
    """Profile a caller asked for through DIR/profile.override ({"profile":"debug","pid":N,"until":epoch}); None = no request.
    Active while its pid is alive and its `until` has not passed, so a crashed bench cannot leave the strict profile on."""
    try:
        d = json.load(open(os.path.join(DIR, "profile.override"), encoding="utf-8"))
        if d.get("profile") not in PROFILES: return None
        if d.get("until") and time.time() > float(d["until"]): return None
        pid = d.get("pid")
        if pid and ps("@(Get-Process -Id %d -ErrorAction SilentlyContinue).Count" % int(pid)) in ("", "0"): return None
        return d["profile"]
    except Exception:
        return None


def apply_profile(a, prof, explicit=()):
    p = PROFILES[prof]
    a.min_free_gib, a.min_ram_gib, a.min_commit_gib, a.samples = p["free"], p["ram"], p["commit"], p["samples"]
    a.soft_ram_gib, a.page_in = p["soft"], p["page_in"]
    a.profile = prof


def watch(name, a, grace=600, idle_exit=300):
    match = a.match
    base_prof, last_ov = a.profile, 0.0
    jwrite(ev="start", name=name, pid=os.getpid(), boot=boot_id(), profile=a.profile, floor_gib=a.min_free_gib,
           ram_floor_gib=a.min_ram_gib, commit_floor_gib=a.min_commit_gib, samples=a.samples)
    t0 = last_seen = last_check = time.time()
    lim = Limits(a)
    lastw = (0.0, 0.0, 0.0)  # (used_gib, time, ram_gib) of the last journal sample
    for used, avail, cfree, pin in sample_stream():
        now = time.time()
        if now - last_ov >= 3:  # a bench or OpenFT asks for the strict profile through profile.override
            last_ov = now
            want = override_profile() or base_prof
            if want != a.profile:
                old = a.profile; apply_profile(a, want); lim.low = 0
                jwrite(ev="profile", name=name, was=old, now=want, floor_gib=a.min_free_gib,
                       ram_floor_gib=a.min_ram_gib, commit_floor_gib=a.min_commit_gib, samples=a.samples)
        lastw = jsample(name, a, lastw, used, avail, cfree, now, pin)
        why = lim.check(used, avail, cfree, pin)
        if why:
            kill_ft(match, name, why)
            lim.low = 0
        if now - last_check >= 30:  # is any FreeToken python process still alive?
            last_check = now
            n = ps("@(Get-CimInstance Win32_Process | ? { $_.CommandLine -match '%s' -and "
                   "$_.Name -match 'python' -and $_.CommandLine -notmatch 'gpu-guard' }).Count" % match)
            if n not in ("", "0"):
                last_seen = now
            elif now - t0 > grace and now - last_seen > idle_exit:
                break
    jwrite(ev="end", name=name, how="ok (FreeToken gone)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="unnamed")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--clear")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", choices=sorted(PROFILES), default=os.environ.get("FT_GUARD_PROFILE", "debug"),
                    help="debug (strict, default) or normal (everyday serving; fires less)")
    prof = PROFILES[ap.parse_known_args()[0].profile]
    ap.add_argument("--min-free-gib", type=float,
                    default=float(os.environ.get("FT_GUARD_MIN_FREE_GIB", prof["free"])))
    ap.add_argument("--min-ram-gib", type=float,
                    default=float(os.environ.get("FT_GUARD_MIN_RAM_GIB", prof["ram"])))
    ap.add_argument("--soft-ram-gib", type=float, default=float(os.environ.get("FT_GUARD_SOFT_RAM_GIB", prof["soft"])),
                    help="available RAM under this only counts while paging in >= --page-in pages/s (0 = off)")
    ap.add_argument("--page-in", type=float, default=float(os.environ.get("FT_GUARD_PAGE_IN", prof["page_in"])))
    ap.add_argument("--min-commit-gib", type=float,
                    default=float(os.environ.get("FT_GUARD_MIN_COMMIT_GIB", prof["commit"])))
    ap.add_argument("--samples", type=int, default=int(os.environ.get("FT_GUARD_SAMPLES", prof["samples"])),
                    help="samples in a row (1 s apart) under a floor before the kill")
    ap.add_argument("--stale", default=os.environ.get("FT_GUARD_STALE", "--port 1925"),
                    help="wrap mode: kill leftover processes whose command line has this before the run "
                         "(default: an orphaned probe server); '' turns it off")
    ap.add_argument("--low-prio", action="store_true", help="wrap mode: run CMD at below-normal priority")
    ap.add_argument("--total-gib", type=float,
                    default=float(os.environ.get("FT_GUARD_TOTAL_GIB", "16")))
    ap.add_argument("--match", default=os.environ.get("FT_GUARD_MATCH", "freetoken"))
    ap.add_argument("--timeout", type=int, default=0, help="wrap mode: seconds, 0 = none")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    if a.clear:
        k = killers(); k.pop(a.clear, None); save_killers(k); print("cleared", a.clear); return
    resolve_unclean()
    if a.report:
        print(json.dumps(killers(), indent=1)); return
    if a.check:
        k = killers()
        if a.name in k and not a.force:
            print(f"[gpu-guard] '{a.name}' is a suspected PC-killer: {k[a.name]}\n"
                  f"[gpu-guard] Not launching. Override: gpu-guard.py --clear {a.name}")
            sys.exit(3)
        return
    if a.watch:
        watch(a.name, a)
        return
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("give --check, --watch, --report, --clear or a command after --")
    if a.name in killers() and not a.force:
        print(f"[gpu-guard] '{a.name}' is a suspected killer. Skipped (--force to run)."); sys.exit(3)
    stale = lambda: [p["ProcessId"] for p in proc_list(a.stale) if "python" in p["CommandLine"].lower().split(" ")[0]] if a.stale else []
    old = stale()
    if old:  # an orphaned server from an earlier run would double the load (the 2026-11-07 07:21 crash)
        print(f"[gpu-guard] killing leftover processes {old} ('{a.stale}')", flush=True)
        jwrite(ev="kill", name=a.name, why="leftover before run", pids=old)
        kill_pids(old)
        print(f"[gpu-guard] leftovers gone in {wait_recovered(old, float("inf"), timeout=30)} s", flush=True)
    base = next(sample_stream())[0] / GIB
    prio = 0x00004000 if a.low_prio else 0  # BELOW_NORMAL_PRIORITY_CLASS (children inherit it, skews tok/s)
    proc = subprocess.Popen(cmd, creationflags=prio | 0x00000200)  # new process group
    jwrite(ev="start", name=a.name, pid=os.getpid(), boot=boot_id(), floor_gib=a.min_free_gib,
           ram_floor_gib=a.min_ram_gib, commit_floor_gib=a.min_commit_gib, base_vram_gib=round(base, 2), cmd=cmd)
    killed = []

    def watcher():
        def kill_tree(why):
            killed.append(why)
            pids = [proc.pid] + [p for p in stale() if p != proc.pid]  # the tree, plus a server that lost its parent
            tip = kill_hint(why)
            jwrite(ev="kill", name=a.name, why=why, pids=pids, **({"tip": tip} if tip else {}))
            print(f"[gpu-guard] KILL '{a.name}': {why}", flush=True)
            note_stop(a.name, why)
            kill_pids(pids)
            s = wait_recovered(pids, base)
            jwrite(ev="recovered", name=a.name, secs=s)
            print(f"[gpu-guard] recovered in {s} s" if s is not None else
                  "[gpu-guard] NOT recovered after 45 s: check for leftover python processes", flush=True)
        t0, lim, lastw = time.time(), Limits(a), (0.0, 0.0, 0.0)
        for used, avail, cfree, pin in sample_stream():
            now = time.time()
            lastw = jsample(a.name, a, lastw, used, avail, cfree, now, pin)
            why = lim.check(used, avail, cfree, pin)
            if not why and a.timeout and now - t0 > a.timeout:
                why = f"timeout {a.timeout}s"
            if why:
                kill_tree(why)
            if proc.poll() is not None or killed:
                return
    th = threading.Thread(target=watcher, daemon=True)
    th.start()
    rc = proc.wait()
    th.join(60)  # let a kill finish its recovery check before the next run starts
    left = stale()
    if left:  # the probe exited but its server did not
        jwrite(ev="kill", name=a.name, why="leftover after run", pids=left)
        kill_pids(left)
    jwrite(ev="end", name=a.name, how=killed[0] if killed else "ok", rc=rc)
    sys.exit(4 if killed else rc)


if __name__ == "__main__":
    main()
