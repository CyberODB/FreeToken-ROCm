"""Drive ft-speedprobe with a mixed real-text workload and snapshot FT_EXPERT_FREQ after each request."""
import importlib.util, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FREQ = os.path.join(HERE, "freq.json")
sp = importlib.util.spec_from_file_location("probe", r"E:\freetoken-021\scripts\ft-speedprobe.py")
probe = importlib.util.module_from_spec(sp); sp.loader.exec_module(probe)

doc = open(r"T:\robot\docs\18-freetoken-rocm-speed-plan.md", encoding="utf-8", errors="replace").read()
MIX = [
    ("code", "Write a Python function that merges overlapping intervals, with type hints, a docstring and three unit tests.", 400),
    ("prose", "Write a short story opening about a lighthouse keeper who finds a message in a bottle.", 400),
    ("math", "Explain step by step why the sum of the first n odd numbers is n squared, then prove it by induction.", 400),
    ("sysadmin", "My Windows service fails to start with error 1053. Walk me through diagnosing it with PowerShell commands.", 400),
    ("chat", "I'm deciding between repairing my old car and buying a used one. What questions should I ask myself?", 400),
    ("sql", "Design a PostgreSQL schema for a library lending system and write a query for the ten most overdue books.", 400),
    ("longdoc-a", doc[:20000] + "\n\nSummarize the document above in about 250 words.", 350),
    ("longdoc-b", doc[20000:60000] + "\n\nList the ten most important measured results in the text above.", 350),
    ("flush", "Say OK.", 4),
]
orig = probe.chat
state = {"n": 0}


def chat(port, prompt, max_tokens, timeout=900):
    state["n"] += 1
    if state["n"] == 1:  # the probe's warm call
        return orig(port, prompt, max_tokens, timeout)
    r = None
    for i, (name, p, mt) in enumerate(MIX):
        r = orig(port, p, mt, timeout)
        print(f"\n    {name}: prompt {r['prompt_tokens']} prefill {r['prefill_tps']} tok/s decode {r['decode_tps']} tok/s ({r['gen_tokens']} tok)", flush=True)
        if os.path.exists(FREQ):  # holds everything up to the start of this request
            shutil.copyfile(FREQ, os.path.join(HERE, f"freq-before-{i:02d}-{name}.json"))
    return r


probe.chat = chat
sys.argv = ["ft-speedprobe.py", "freq-mix", "--env", f"FT_EXPERT_FREQ={FREQ}",
            "--args", "--moe-strategy hybrid --memory-ratio 0.75", "--tests", "decode"]
probe.main()
