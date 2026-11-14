"""Decode test with the daemon's settings (default memory ratio) and an optional desktop VRAM reserve.
usage: vrrun.py NAME RESERVE(none|auto|GiB) [TESTS]"""
import importlib.util, subprocess, sys

name, reserve = sys.argv[1], sys.argv[2]
sp = importlib.util.spec_from_file_location("probe", r"E:\freetoken-021\scripts\ft-speedprobe.py")
probe = importlib.util.module_from_spec(sp); sp.loader.exec_module(probe)
PS = ("$s = (Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage','\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples; "
      "$d = ($s | Where-Object Path -like '*dedicated*' | Measure-Object CookedValue -Maximum).Maximum; "
      "$h = ($s | Where-Object Path -like '*shared*' | Measure-Object CookedValue -Maximum).Maximum; "
      "'gpu_dedicated_mb=' + [math]::Round($d/1MB) + ' gpu_shared_mb=' + [math]::Round($h/1MB)")
orig = probe.chat


def chat(port, prompt, max_tokens, timeout=900):
    r = orig(port, prompt, max_tokens, timeout)
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS], capture_output=True, text=True).stdout.strip()
    print(f"\n    VR decode_tps={r['decode_tps']} prefill_tps={r['prefill_tps']} {out}", flush=True)
    return r


probe.chat = chat
env = ["--env", "FREETOKEN_PIN_BUDGET_GB=2", "--env", "FREETOKEN_MAP_LOCKED_BANKS=1", "--env", "FREETOKEN_SKIP_BANK_LOCK=1"]
if reserve != "none":
    env += ["--env", f"FREETOKEN_DESKTOP_RESERVE_GB={reserve}"]
sys.argv = ["ft-speedprobe.py", name, *env, "--args", "--moe-strategy hybrid", "--tests", sys.argv[3] if len(sys.argv) > 3 else "decode"]
probe.main()
