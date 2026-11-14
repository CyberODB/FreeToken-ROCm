"""Decode before / during / after a RAM balloon, printing server working set and available RAM.
usage: pressrun.py NAME PIN_GB NOLOCK(0|1) BALLOON_GIB"""
import importlib.util, subprocess, sys

name, pin, nolock, gib = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sp = importlib.util.spec_from_file_location("probe", r"E:\freetoken-021\scripts\ft-speedprobe.py")
probe = importlib.util.module_from_spec(sp); sp.loader.exec_module(probe)
PS = ("$p = Get-Process python | Sort-Object VirtualMemorySize64 -Descending | Select-Object -First 1; "
      "$o = Get-CimInstance Win32_OperatingSystem; "
      "'ws_gib=' + [math]::Round($p.WorkingSet64/1GB,2) + ' avail_gib=' + [math]::Round($o.FreePhysicalMemory/1MB,2)")
orig = probe.chat
state = {"n": 0}


def mem():
    return subprocess.run(["powershell", "-NoProfile", "-Command", PS], capture_output=True, text=True).stdout.strip()


def step(tag, port, prompt, max_tokens, timeout):
    r = orig(port, prompt, max_tokens, timeout)
    print(f"\n    PRESS {tag}: decode_tps={r['decode_tps']} ttft={r.get('ttft')} {mem()}", flush=True)
    return r


def chat(port, prompt, max_tokens, timeout=900):
    state["n"] += 1
    if state["n"] == 1:
        return orig(port, prompt, max_tokens, timeout)
    step("before", port, prompt, max_tokens, timeout)
    b = subprocess.Popen([sys.executable, r"E:\freetoken-021\speedprobe\balloon.py", gib], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    b.stdout.readline()
    print(f"\n    PRESS balloon {gib} GiB up: {mem()}", flush=True)
    try:
        step("during-1", port, prompt, max_tokens, timeout)
        step("during-2", port, prompt, max_tokens, timeout)
    finally:
        b.stdin.close(); b.wait(timeout=60)
    return step("after", port, prompt, max_tokens, timeout)


probe.chat = chat
env = ["--env", f"FREETOKEN_PIN_BUDGET_GB={pin}", "--env", "FREETOKEN_MAP_LOCKED_BANKS=1"]
if nolock == "1":
    env += ["--env", "FREETOKEN_SKIP_BANK_LOCK=1"]
sys.argv = ["ft-speedprobe.py", name, *env, "--args", "--moe-strategy hybrid --memory-ratio 0.75", "--tests", "decode"]
probe.main()
