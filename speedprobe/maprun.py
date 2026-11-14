"""Run ft-speedprobe's decode test at one pin budget and print host RAM figures while the server is still up.
usage: maprun.py NAME PIN_GB MAP(0|1)"""
import importlib.util, subprocess, sys

name, pin, mp = sys.argv[1], sys.argv[2], sys.argv[3]
sp = importlib.util.spec_from_file_location("probe", r"E:\freetoken-021\scripts\ft-speedprobe.py")
probe = importlib.util.module_from_spec(sp); sp.loader.exec_module(probe)
PS = ("$p = Get-Process python | Sort-Object WorkingSet64 -Descending | Select-Object -First 1; "
      "$o = Get-CimInstance Win32_OperatingSystem; "
      "'MEM ws_gib=' + [math]::Round($p.WorkingSet64/1GB,2) + ' private_gib=' + [math]::Round($p.PrivateMemorySize64/1GB,2) + "
      "' avail_gib=' + [math]::Round($o.FreePhysicalMemory/1MB,2)")
orig = probe.chat


def chat(port, prompt, max_tokens, timeout=900):
    r = orig(port, prompt, max_tokens, timeout)
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS], capture_output=True, text=True).stdout.strip()
    print(f"\n    {out} decode_tps={r['decode_tps']}", flush=True)
    return r


probe.chat = chat
env = ["--env", f"FREETOKEN_PIN_BUDGET_GB={pin}"]
if mp == "1":
    env += ["--env", "FREETOKEN_MAP_LOCKED_BANKS=1"]
sys.argv = ["ft-speedprobe.py", name, *env, "--args", "--moe-strategy hybrid --memory-ratio 0.75", "--tests", sys.argv[4] if len(sys.argv) > 4 else "decode"]
probe.main()
