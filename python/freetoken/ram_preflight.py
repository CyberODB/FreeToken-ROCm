"""RAM check before a model load (ROCm fork, Windows, 2026-11-06).

Pinned and OS-locked expert banks cannot be paged out, so when they do not fit in free RAM
Windows pages the desktop instead. Before ``ft serve`` loads anything this compares the
model's host-RAM need with free RAM. If it is short, a small window explains why, lists the
programs using the most memory with a Close button each, and lets you continue (or cancel).
It warns and never refuses: Continue, closing the window or the timeout all load the model.

On only when FREETOKEN_HOST_RESERVE_GB is set (the launcher sets it from ram-baseline.py):
  FREETOKEN_HOST_RESERVE_GB   RAM the desktop needs (measured baseline + margin)
  FREETOKEN_HOST_MARGIN_GB    free RAM to keep spare after the load (default 2); the window opens below that
  FREETOKEN_HOST_EXTRA_GB     allowance for the serve process beyond the weights (default 1.5, a guess)
  FREETOKEN_PREFLIGHT_WAIT_S  seconds before the window continues by itself (default 120)
Need = FTW bank bytes, else the model's weight file sizes (an upper bound: GGUF files hold
non-expert weights too), plus the allowance. Exit 0 = load, 1 = cancelled.
Standard library + tkinter only, run as its own process: ``python -m freetoken.ram_preflight <model>``.
"""
from __future__ import annotations

import csv
import ctypes
import io
import os
import subprocess
import sys
import time

GIB = 1 << 30
LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "ram-preflight.log")
# never offered for closing: Windows itself, security, and the AI stack this load belongs to
SYSTEM = {
    "system", "registry", "smss.exe", "csrss.exe", "wininit.exe", "services.exe", "lsass.exe", "winlogon.exe",
    "svchost.exe", "dwm.exe", "explorer.exe", "memory compression", "secure system", "fontdrvhost.exe",
    "sihost.exe", "ctfmon.exe", "audiodg.exe", "msmpeng.exe", "nissrv.exe", "searchhost.exe", "searchindexer.exe",
    "startmenuexperiencehost.exe", "shellexperiencehost.exe", "textinputhost.exe", "runtimebroker.exe",
    "taskhostw.exe", "conhost.exe", "cmd.exe", "dllhost.exe", "spoolsv.exe", "wudfhost.exe", "lsaiso.exe",
    "python.exe", "pythonw.exe", "py.exe", "ft.exe", "wsl.exe", "wslservice.exe", "wslhost.exe", "vmmem",
    "vmmemwsl", "ollama.exe", "ollama app.exe", "llama-server.exe", "tasklist.exe", "taskkill.exe",
}
MIN_GB = 0.2  # smaller programs are not worth listing
# shown so you can see them, never given a Close button: a leftover serve, a bench, Ollama, WSL, shells, Claude, WebView2
INFO_ONLY = {"python.exe", "pythonw.exe", "ollama.exe", "ollama app.exe", "llama-server.exe", "vmmem", "vmmemwsl",
             "wslservice.exe", "wslhost.exe", "pwsh.exe", "powershell.exe", "windowsterminal.exe", "msedgewebview2.exe",
             "claude.exe"}


class _MS(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ("tphys", "aphys", "tpage", "apage", "tvirt", "avirt", "ext")]


def ram() -> tuple[float, float]:
    """(total, available) in GiB."""
    m = _MS(); m.len = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.tphys / GIB, m.aphys / GIB


def model_need_gib(model: str) -> tuple[float, str]:
    """(GiB, how it was worked out) for the weights' host RAM."""
    meta = os.path.join(model, "freetoken_weight.json")  # same read as expert_banks.ftw_bank_bytes, without importing torch
    if os.path.isfile(meta):
        try:
            import json
            with open(meta, encoding="utf-8") as f:
                b = sum(t["nbytes"] for t in json.load(f).get("tensors", []) if t.get("kind") == "experts_bank")
            if b:
                return b / GIB, "FTW expert banks"
        except Exception:
            pass
    if os.path.isfile(model):
        if model.lower().endswith(".gguf"):
            try:
                return gguf_expert_bytes(model) / GIB, "GGUF expert tensors"
            except Exception as exc:
                log(f"GGUF header read failed, using file size: {exc!r}")
        return os.path.getsize(model) / GIB, "model file size"
    total = 0
    for root, _, files in os.walk(model):
        for f in files:
            if f.endswith((".safetensors", ".gguf", ".bin")) and "mmproj" not in f.lower():
                total += os.path.getsize(os.path.join(root, f))
    return total / GIB, "weight file sizes"


def gguf_expert_bytes(path: str) -> int:
    """Bytes of the routed-expert tensors (``*_exps``), the part held in host RAM, read from the GGUF header only.

    Each tensor's size is the gap to the next tensor's offset (the last runs to the end of the file)."""
    import struct

    with open(path, "rb") as f:
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        rd("I")  # version
        n_tensors, n_kv = rd("Q"), rd("Q")
        string = lambda: f.read(rd("Q")).decode("utf-8", "replace")
        fixed = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = rd("I"), rd("Q")
                if et in fixed:
                    f.seek(n * struct.calcsize(fixed[et]), 1)
                else:
                    for _ in range(n):
                        value(et)
                return None
            return rd(fixed[t])

        align = 32
        for _ in range(n_kv):
            key = string()
            v = value(rd("I"))
            if key == "general.alignment":
                align = int(v)
        infos = []
        for _ in range(n_tensors):
            name = string()
            f.seek(8 * rd("I"), 1)  # dims
            rd("I")  # ggml type
            infos.append((rd("Q"), name))
        data = -(-f.tell() // align) * align
    end = os.path.getsize(path) - data
    infos.sort()
    sizes = [(nxt[0] if nxt else end) - off for (off, _), nxt in zip(infos, infos[1:] + [None])]
    return sum(s for s, (_, name) in zip(sizes, infos) if "_exps" in name)


def _parents() -> dict[int, int]:
    """pid -> parent pid, from a Toolhelp snapshot."""
    from ctypes import wintypes

    class PE(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("pid", wintypes.DWORD),
                    ("heap", ctypes.c_size_t), ("mod", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("ppid", wintypes.DWORD), ("prio", ctypes.c_long), ("flags", wintypes.DWORD),
                    ("exe", ctypes.c_char * 260)]
    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    out, e = {}, PE()
    e.dwSize = ctypes.sizeof(e)
    ok = k32.Process32First(snap, ctypes.byref(e))
    while ok:
        out[e.pid] = e.ppid
        ok = k32.Process32Next(snap, ctypes.byref(e))
    k32.CloseHandle(snap)
    return out


def _ancestors() -> set[int]:
    """This check, the serve and everything that started it (daemon, desktop app): never closed."""
    try:
        par = _parents()
    except Exception:
        return {os.getpid(), os.getppid()}
    seen, pid = set(), os.getpid()
    while pid and pid not in seen:
        seen.add(pid)
        pid = par.get(pid, 0)
    return seen


def programs() -> list[dict]:
    """Programs using at least MIN_GB, biggest first: name, pids, GiB working set, and whether Close is offered.

    The AI stack, Claude, shells and WebView2 (FreeToken Desktop's own window may be one) are listed
    for information only. Windows' own processes are left out."""
    out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    keep = _ancestors()
    g: dict[str, dict] = {}
    for row in csv.reader(io.StringIO(out)):
        if len(row) < 5 or not row[1].isdigit():
            continue
        name, pid = row[0], int(row[1])
        low = name.lower()
        if low in SYSTEM and low not in INFO_ONLY:
            continue
        kb = int("".join(c for c in row[4] if c.isdigit()) or 0)
        e = g.setdefault(name, {"name": name, "pids": [], "gib": 0.0, "ancestor": False})
        e["gib"] += kb / 1048576
        if pid in keep:
            e["ancestor"] = True
        else:
            e["pids"].append(pid)
    rows = [e for e in g.values() if e["gib"] >= MIN_GB]
    for e in rows:
        low = e["name"].lower()
        e["closable"] = bool(e["pids"]) and not e["ancestor"] and low not in INFO_ONLY and not low.startswith(("freetoken", "claude"))
    return sorted(rows, key=lambda e: -e["gib"])


def close(pids: list[int], force: bool = False) -> None:
    """Ask those processes to quit (force: kill them). By PID, never by name, and not their child trees."""
    args = ["taskkill"] + [a for p in pids for a in ("/pid", str(p))] + (["/f"] if force else [])
    subprocess.run(args, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def log(line: str) -> None:
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
    except OSError:
        pass


def check(model: str) -> int:
    reserve = float(os.environ["FREETOKEN_HOST_RESERVE_GB"])
    extra = float(os.environ.get("FREETOKEN_HOST_EXTRA_GB", "1.5"))
    wait = int(float(os.environ.get("FREETOKEN_PREFLIGHT_WAIT_S", "120")))
    margin = float(os.environ.get("FREETOKEN_HOST_MARGIN_GB", "2"))
    weights, how = model_need_gib(model)
    need = weights + extra
    total, avail = ram()
    budget = total - reserve
    short = need + margin - avail
    log(f"{os.path.basename(model)}: need {need:.1f} GiB ({how} {weights:.1f} + {extra:g}), "
        f"free {avail:.1f}, margin {margin:g}, budget {budget:.1f} (total {total:.1f} - reserve {reserve:g})")
    if short <= 0:
        return 0
    return window(model, need, how, weights, extra, budget, wait, margin)


def window(model, need, how, weights, extra, budget, wait, margin) -> int:
    import tkinter as tk

    result = {"code": 0}
    root = tk.Tk()
    root.title("FreeToken: low RAM before load")
    root.attributes("-topmost", True)
    head = tk.Label(root, justify="left", anchor="w", font=("Segoe UI", 10))
    head.pack(fill="x", padx=12, pady=(10, 4))
    tk.Label(root, anchor="w", font=("Segoe UI", 9, "italic"), justify="left",
             text="Close programs you don't need, then press Continue. Close asks the program to quit;\n"
                  "a second press forces it. Rows marked 'info only' (Claude, shells, Python, Ollama, WSL)\n"
                  "are shown so you can see them; close those yourself if you mean to.").pack(fill="x", padx=12)
    box = tk.Frame(root)
    box.pack(fill="both", padx=12, pady=6)
    bar = tk.Frame(root)
    bar.pack(fill="x", padx=12, pady=(0, 10))
    timer = tk.Label(bar, anchor="w")
    timer.pack(side="left")
    asked: set[str] = set()
    deadline = [time.monotonic() + wait]

    def refresh():
        total, avail = ram()
        short = need + margin - avail
        fit = "fits" if need <= budget else f"is {need - budget:.1f} GiB over even with only your usual desktop open"
        state = (f"Short by {short:.1f} GiB (keeping {margin:g} GiB spare). Loading now will push the desktop into the pagefile."
                 if short > 0 else "Enough free RAM now. Continue to load.")
        head.config(text=(
            f"{os.path.basename(model)}\n"
            f"Needs about {need:.1f} GiB of RAM ({how} {weights:.1f} + {extra:g} allowance), held locked: Windows cannot page it out.\n"
            f"Free right now: {avail:.1f} GiB.  {state}\n"
            f"Budget beside your usual desktop: {budget:.1f} GiB, so this model {fit}."))
        for w in box.winfo_children():
            w.destroy()
        for i, p in enumerate(programs()[:14]):
            n = len(p["pids"]) + p["ancestor"]
            tk.Label(box, text=p["name"] + (f"  x{n}" if n > 1 else ""), anchor="w", width=32).grid(row=i, column=0, sticky="w")
            tk.Label(box, text=f"{p['gib']:.2f} GiB", width=10, anchor="e").grid(row=i, column=1)
            if p["closable"]:
                label = "Force close" if p["name"] in asked else "Close"
                tk.Button(box, text=label, width=11, command=lambda q=p: do_close(q)).grid(row=i, column=2, padx=4, pady=1)
            else:
                why = "this load" if p["ancestor"] else "info only"
                tk.Label(box, text=why, width=11, fg="gray40").grid(row=i, column=2, padx=4)

    def do_close(p):
        force = p["name"] in asked
        asked.add(p["name"])
        close(p["pids"], force)
        log(f"{'force-' if force else ''}closed {p['name']} pids {p['pids']}")
        root.after(2500, refresh)

    def finish(code):
        result["code"] = code
        log("cancelled" if code else "continued")
        root.destroy()

    def tick():
        left = int(deadline[0] - time.monotonic())
        if left <= 0:
            finish(0)
            return
        timer.config(text=f"Continues by itself in {left} s")
        root.after(1000, tick)

    tk.Button(bar, text="Continue loading", width=16, command=lambda: finish(0)).pack(side="right")
    tk.Button(bar, text="Cancel load", width=12, command=lambda: finish(1)).pack(side="right", padx=6)
    tk.Button(bar, text="Refresh", width=9, command=refresh).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", lambda: finish(0))
    refresh()
    tick()
    root.mainloop()
    return result["code"]


def run_before_serve(argv: list[str]) -> bool:
    """Called by ``ft serve``: True = go on loading. Silent no-op off Windows or without the env var."""
    if os.name != "nt" or not os.environ.get("FREETOKEN_HOST_RESERVE_GB"):
        return True
    model = None
    for i, a in enumerate(argv):
        if a in ("--model", "--model-path") and i + 1 < len(argv):
            model = argv[i + 1]
        elif a.startswith(("--model=", "--model-path=")):
            model = a.split("=", 1)[1]
    if not model or not os.path.exists(model):
        return True  # a hub id or a typo: the serve reports it, nothing to size here
    try:
        code = subprocess.run([sys.executable, "-m", "freetoken.ram_preflight", model]).returncode
    except Exception as exc:  # a broken check must never block a load
        log(f"preflight failed: {exc!r}")
        return True
    return code != 1


if __name__ == "__main__":
    try:
        sys.exit(check(sys.argv[1]))
    except Exception as exc:
        log(f"preflight error: {exc!r}")
        sys.exit(0)
