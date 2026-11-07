"""GPU guard for FreeToken tests on Windows: VRAM floor, kill switch, crash journal.

The desktop launcher runs these two steps for every test:
  gpu-guard.py --check --name NAME     exit 3 if NAME is a suspected PC-killer (launcher stops)
  gpu-guard.py --watch --name NAME     background watcher (hidden), runs while FreeToken runs
Other uses:
  gpu-guard.py --report                journal summary and the suspected-killer list
  gpu-guard.py --clear NAME            forget a suspected killer
  gpu-guard.py --name N [opts] -- CMD  run one command under the same rules (kills CMD's tree)

Rules:
  * Free dedicated VRAM under --min-free-gib for 2 samples in a row (2 s apart) kills the
    FreeToken GPU processes (python/ft processes whose command line mentions "freetoken" and
    that hold VRAM), so the desktop and a playing video keep memory. Browsers etc. are never touched.
  * Every VRAM sample is written to the journal and fsynced, so the last line before a freeze
    survives a hard reset.
  * A watcher session with no "end" line after a reboot is looked at: if Windows logged
    Kernel-Power event 41 (unclean shutdown) after the last sample, that test is a suspected
    killer and --check refuses it until --force or --clear. A clean restart is not blamed.
Journal and killer list: $FT_GUARD_DIR (default %LOCALAPPDATA%\\freetoken-gpu-guard).
Env: FT_GUARD_MIN_FREE_GIB (default 1.0), FT_GUARD_TOTAL_GIB (default 16), FT_GUARD_MATCH (default "freetoken").
"""
import argparse, csv, io, json, os, subprocess, sys, threading, time

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


def vram_stream():
    """Yield used dedicated bytes of the busiest adapter every 2 s."""
    p = subprocess.Popen(["typeperf", r"\GPU Adapter Memory(*)\Dedicated Usage", "-si", "2"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         creationflags=NOWIN)
    try:
        for line in p.stdout:
            parts = [x.strip('"') for x in line.strip().split(",")]
            try:
                yield max(float(x) for x in parts[1:])
            except ValueError:
                continue
    finally:
        p.kill()


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
            "select ProcessId,CommandLine | ConvertTo-Json -Compress")
    procs = json.loads(js or "[]")
    procs = procs if isinstance(procs, list) else [procs]
    hits = []
    for p in procs:
        c = p["CommandLine"].lower()
        if (match.lower() in c and "gpu-guard" not in c and p["ProcessId"] != me
                and used.get(p["ProcessId"], 0) > 0):
            hits.append((p["ProcessId"], used[p["ProcessId"]]))
    return hits


def kill_ft(match, name, why):
    hits = ft_gpu_pids(match)
    jwrite(ev="kill", name=name, why=why, pids=[h[0] for h in hits])
    print(f"[gpu-guard] KILL '{name}': {why}; pids {[h[0] for h in hits]}", flush=True)
    for pid, _ in hits:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                       creationflags=NOWIN)
    return bool(hits)


def watch(name, floor, total, match, grace=600, idle_exit=300):
    jwrite(ev="start", name=name, pid=os.getpid(), boot=boot_id(), floor_gib=floor)
    t0 = last_seen = last_check = time.time()
    low = 0
    lastw = (0.0, 0.0)  # (used_gib, time) of the last journal sample
    for used in vram_stream():
        now = time.time()
        used_g = used / GIB
        free = total - used_g
        if abs(used_g - lastw[0]) > 0.05 or now - lastw[1] >= 10 or free < floor + 1:
            jwrite(ev="vram", name=name, used_gib=round(used_g, 2), free_gib=round(free, 2))
            lastw = (used_g, now)
        low = low + 1 if free < floor else 0
        if low >= 2:
            kill_ft(match, name, f"free VRAM {free:.2f} GiB under floor {floor}")
            low = 0
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
    ap.add_argument("--min-free-gib", type=float,
                    default=float(os.environ.get("FT_GUARD_MIN_FREE_GIB", "1.0")))
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
        watch(a.name, a.min_free_gib, a.total_gib, a.match)
        return
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("give --check, --watch, --report, --clear or a command after --")
    if a.name in killers() and not a.force:
        print(f"[gpu-guard] '{a.name}' is a suspected killer. Skipped (--force to run)."); sys.exit(3)
    proc = subprocess.Popen(cmd, creationflags=0x00004000 | 0x00000200)  # below normal, new group
    jwrite(ev="start", name=a.name, pid=os.getpid(), boot=boot_id(), floor_gib=a.min_free_gib, cmd=cmd)
    killed = []

    def watcher():
        def kill_tree(why):
            killed.append(why)
            jwrite(ev="kill", name=a.name, why=why)
            print(f"[gpu-guard] KILL '{a.name}': {why}", flush=True)
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                           creationflags=NOWIN)
        t0, low = time.time(), 0
        for used in vram_stream():
            free = a.total_gib - used / GIB
            jwrite(ev="vram", name=a.name, used_gib=round(used / GIB, 2), free_gib=round(free, 2))
            low = low + 1 if free < a.min_free_gib else 0
            if low >= 2:
                kill_tree(f"free VRAM {free:.2f} GiB under floor {a.min_free_gib}")
            elif a.timeout and time.time() - t0 > a.timeout:
                kill_tree(f"timeout {a.timeout}s")
            if proc.poll() is not None or killed:
                return
    threading.Thread(target=watcher, daemon=True).start()
    rc = proc.wait()
    jwrite(ev="end", name=a.name, how=killed[0] if killed else "ok", rc=rc)
    sys.exit(4 if killed else rc)


if __name__ == "__main__":
    main()
