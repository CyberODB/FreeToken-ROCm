"""ft-speedprobe: one FreeToken ROCm launch, measured. Added 2026-11-06 for the speed plan (T:\\robot\\docs\\18).

  py -u ft-speedprobe.py NAME [--env K=V ...] [--args "..."] [--model PATH] [--tests decode,p5k,p25k] [--port 1925]

Starts `bin\\ft.cmd serve` itself (not the daemon), so each run gets its own env (pin budget, stamps, threads).
Pauses ft-router (:1919) while it runs, as bench.py does; the router resumes when this process exits.
Refuses to start if the daemon, the router's backend or Ollama holds a model.

Tests (streamed, temp 0, thinking off):
  decode  short prompt, 512 tokens, twice; the second run counts (the first warms the slot cache)
  p5k     ~5K-token prompt: prefill tok/s = prompt tokens / TTFT, then 256 tokens of decode at that depth
  p25k    same at ~25K tokens
  cN      (c2, c3, c4, ...) N concurrent requests at once, N different essay prompts, 512 tokens each; decode_tps is the
          TOTAL throughput (all generated tokens / wall time from first token to last token), per_req_tps the single streams
Each test also keeps the [graph-stamps] lines it produced, when FREETOKEN_GRAPH_STAMPS is set.
The decode warm-up (the first request, so the first replay of each graph) is kept as tests.decode_warm, and the
graph.py probes are collected into res["probes"]: [alloc-probe] lines and [graph-upload] ok/failed counts (doc 27 A/B).
Results: one JSON line per run in speedprobe\\results.jsonl next to this folder; server log in speedprobe\\NAME.log.
"""
import argparse, json, os, random, re, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
OUT = os.path.join(SRC, "speedprobe")
MODEL = "E:/ai-models/ft-models/HauhauCS_Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_NL.gguf"
sys.path.insert(0, HERE)
from ft_daemon_env import daemon_env  # 2026-11-14: the daemon launcher's env, not a fixed pin 14 / graph bs 1. Undo: *.bak-20261114-benchsafe
BASE_ENV = daemon_env()
BASE_ARGS = ["--moe-cpu-layers", "auto", "--moe-cache-size", "5000"]
ESSAY = "Write a detailed 900-word essay on the history of the printing press."
LOAD_RX = re.compile(r"Free memory|moe-cpu-layers|expert banks|CPU MoE executor|stream memops|max_total_num_tokens|context len|Free GPU memory after|slots|Load weight end|error|Error|Traceback")


def http(method, url, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def try_json(method, url, body=None, timeout=3):
    try:
        s, t = http(method, url, body, timeout)
        return s, (json.loads(t) if t.strip().startswith(("{", "[")) else t)
    except Exception as e:
        return None, str(e)


def gpu_busy():
    """Who holds the GPU now, or None."""
    s, st = try_json("GET", "http://127.0.0.1:19000/engine/status")
    if s == 200 and isinstance(st, dict) and (st.get("running") or st.get("starting")):
        return f"the FreeToken daemon is running a model on :{st.get('port')}"
    s, _ = try_json("GET", "http://127.0.0.1:1920/health")
    if s == 200:
        return "ft-router's backend (:1920) has a model loaded"
    s, ps = try_json("GET", "http://127.0.0.1:11434/api/ps")
    if s == 200 and isinstance(ps, dict) and ps.get("models"):
        return "Ollama has a model loaded"
    return None


def make_prompt(n_tokens, seed):
    """Plain prose-like filler of roughly n tokens, unique per run so no prefix cache hits."""
    rnd = random.Random(seed)
    words = ("river stone harbor ledger lantern copper orchard signal margin weaver quarry beacon thistle canal "
             "furnace meadow archive compass timber vessel garnet pillar summit cobalt hollow saffron tundra "
             "granite falcon mortar prism ember willow basalt cipher anchor juniper").split()
    out, n = [f"Reference notes, set {seed}.\n"], 0
    while n < n_tokens:
        k = rnd.randint(8, 18)
        s = " ".join(rnd.choice(words) for _ in range(k))
        out.append(f"{s.capitalize()} near marker {rnd.randint(100, 99999)}. ")
        n += k + 6
        if rnd.random() < 0.08:
            out.append("\n")
    return "".join(out) + "\n\nIn about 200 words, describe what kind of document the notes above could be."


def count_tokens(port, prompt):
    """The prompt's real token count from the server (/v1/messages/count_tokens, same tokenizer + chat template as a
    generation), or None when the server can't say."""
    body = {"model": "probe", "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/messages/count_tokens", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return int(json.load(r)["input_tokens"])
    except Exception:
        return None


def fit_prompt(port, n):
    """make_prompt(n) counts ~1 token per word, which is tokenizer-dependent: LFM2's tokenizer gives ~1.39x, so a "p60k"
    was 83K tokens and overflowed a 64K context (2026-11-13). Count it on the server, rescale once, recount."""
    seed, ask = random.randrange(1 << 30), n
    p = make_prompt(ask, seed)
    for _ in range(3):
        got = count_tokens(port, p)
        if not got or abs(got - n) <= max(64, n // 100):
            break
        ask = max(1, round(ask * n / got))
        p = make_prompt(ask, seed)
    return p


def chat(port, prompt, max_tokens, timeout=900):
    body = {"model": "probe", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = t_last = None; chunks = 0; usage = None; text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices") or []:
                delta = c.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    text.append(delta.get("content") or "")
                    now = time.perf_counter()
                    t_first = t_first or now; t_last = now; chunks += 1
    t_end = time.perf_counter()
    gen = (usage or {}).get("completion_tokens") or chunks
    ptok = (usage or {}).get("prompt_tokens")
    ttft = (t_first or t_end) - t0
    dec = (gen - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first and gen > 1 else None
    return {"prompt_tokens": ptok, "gen_tokens": gen, "ttft_s": round(ttft, 3),
            "prefill_tps": round(ptok / ttft, 1) if ptok and ttft > 0 else None,
            "decode_tps": round(dec, 2) if dec else None, "total_s": round(t_end - t0, 2), "text": "".join(text),
            "t_first": t_first, "t_last": t_last, "t_end": t_end}  # absolute perf_counter times (cN wall-clock window)


ESSAY_TOPICS = ["the printing press", "the steam engine", "the telephone", "the mechanical clock", "the compass",
                "the electric light bulb", "the railway", "the telescope"]


def run_concurrent(port, n, max_tokens, chat_fn=None):
    """cN test: n simultaneous chats with n different essay prompts. Returns the result dict (errors listed, not raised)."""
    chat_fn = chat_fn or chat
    prompts = [f"Write a detailed 900-word essay on the history of {ESSAY_TOPICS[i % len(ESSAY_TOPICS)]}." for i in range(n)]
    out = [None] * n; errs = []
    def work(i):
        try:
            out[i] = chat_fn(port, prompts[i], max_tokens)
        except Exception as e:
            errs.append(f"req {i}: {e}"[:300])
    t_start = time.perf_counter()
    ths = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for th in ths: th.start()
    for th in ths: th.join()
    wall = time.perf_counter() - t_start
    ok = [r for r in out if r]
    gen = sum(r["gen_tokens"] or 0 for r in ok)
    firsts = [r["t_first"] for r in ok if r.get("t_first")]; lasts = [r["t_last"] for r in ok if r.get("t_last")]
    win = (max(lasts) - min(firsts)) if firsts and lasts and max(lasts) > min(firsts) else None
    return {"n": n, "decode_tps": round(gen / win, 2) if win and not errs else None,
            "per_req_tps": [r["decode_tps"] if r else None for r in out], "prefill_tps": None,
            "gen_tokens": gen, "wall_s": round(wall, 2), "errors": errs}


def log_tail(path, start):
    with open(path, "rb") as f:
        f.seek(start)
        return f.read().decode("utf-8", "replace"), f.tell()


def gguf_expert_upper(path):
    """block_count * expert_count from the GGUF header: an upper bound on the model's expert slots (dense leading layers are
    counted too). None when the file is not GGUF or has no expert_count. Stdlib only: reads the key/value header, not tensors."""
    import struct
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    num = {4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
    try:
        with open(path, "rb") as f:
            rd = lambda n: f.read(n)
            u32 = lambda: struct.unpack("<I", rd(4))[0]
            u64 = lambda: struct.unpack("<Q", rd(8))[0]
            if rd(4) != b"GGUF":
                return None
            u32(); u64(); nkv = u64()
            vals = {}

            def skip(t):
                if t in fixed:
                    f.seek(fixed[t], 1)
                elif t == 8:
                    f.seek(u64(), 1)
                elif t == 9:
                    it, n = u32(), u64()
                    if it in fixed:
                        f.seek(fixed[it] * n, 1)
                    else:
                        for _ in range(n):
                            skip(it)
                else:
                    raise ValueError(f"gguf type {t}")

            for _ in range(nkv):
                key = rd(u64()).decode("utf-8", "replace")
                t = u32()
                if key.endswith((".block_count", ".expert_count")) and t in num:
                    vals[key.rsplit(".", 1)[1]] = struct.unpack(num[t], rd(struct.calcsize(num[t])))[0]
                    if len(vals) == 2:
                        return vals["block_count"] * vals["expert_count"]
                else:
                    skip(t)
    except (OSError, ValueError, struct.error):
        return None
    return None


def base_args_for(model, extra):
    """BASE_ARGS, except (2026-11-12) when the model has fewer experts than the fixed 5000 slots: the engine then refuses to
    start ("MoE cache of 5000 slots exceeds what the pools can hold"), e.g. LFM2-24B-A2B with 38 x 64 = 2432 experts. Such a
    model gets --moe-cache-auto, which sizes the slots to the card. A caller's own --moe-cache-size/--moe-cache-auto wins.
    2026-11-13: --moe-strategy fused gets none: the engine refuses --moe-cpu-layers with fused, so every ft-fit stage after a
    fused winner (spec, flags, confirm, context ladder) died at load."""
    if "--moe-strategy=fused" in extra or any(a == "--moe-strategy" and b == "fused" for a, b in zip(extra, extra[1:])):
        return []
    if any(x in extra for x in ("--moe-cache-size", "--moe-cache-auto", "--moe-cache-rate")):
        return ["--moe-cpu-layers", "auto"]
    cap = gguf_expert_upper(model)
    if cap is not None and cap <= int(BASE_ARGS[3]):
        print(f"model has at most {cap} experts, under the {BASE_ARGS[3]}-slot base: using --moe-cache-auto")
        # 2026-11-13: without a KV reserve, --moe-cache-auto keeps only its default 8192-token KV floor, so every context
        # test past ~8K failed ("Input sequence length 8309 exceeds 8279"). Hold back the run's own context instead.
        out = ["--moe-cpu-layers", "auto", "--moe-cache-auto"]
        if "--kv-reserve-tokens" not in extra and "--max-seq-len-override" in extra[:-1]:
            out += ["--kv-reserve-tokens", extra[extra.index("--max-seq-len-override") + 1]]
        return out
    return BASE_ARGS


def stamps_of(text):
    return [re.sub(r"\x1b\[[0-9;]*m", "", l).strip()[:600] for l in text.splitlines() if "[graph-stamps]" in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--env", action="append", default=[], help="K=V, repeatable")
    ap.add_argument("--args", default="", help="extra serve args, one string")
    ap.add_argument("--no-base-args", action="store_true", help="drop the default --moe-cpu-layers auto --moe-cache-size 5000")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tests", default="decode,p5k,p25k")
    ap.add_argument("--port", type=int, default=1925)
    ap.add_argument("--load-timeout", type=int, default=420)
    ap.add_argument("--vision", action="store_true", help="keep the vision encoder (default: --mm-disable vision; all tests are text, vision memory is a fitting question)")
    a = ap.parse_args()

    busy = gpu_busy()
    if busy:
        sys.exit(f"GPU busy: {busy}. Stop it first.")
    os.makedirs(OUT, exist_ok=True)
    logp = os.path.join(OUT, f"{a.name}.log")
    env = dict(os.environ)
    for k, v in BASE_ENV.items():  # the caller's environment and --env win over the daemon's defaults
        env.setdefault(k, v)
    extra_env = dict(kv.split("=", 1) for kv in a.env); env.update(extra_env)
    base_env = {k: env[k] for k in BASE_ENV}
    print(f"[{a.name}] base env (daemon launcher): {' '.join(f'{k}={v}' for k, v in base_env.items())}")
    extra = a.args.split() if a.args else []
    args = ([] if a.no_base_args else base_args_for(a.model, extra)) + extra
    if not a.vision and "--mm-disable" not in args:
        args += ["--mm-disable", "vision"]
    s, _ = try_json("POST", "http://127.0.0.1:1919/router/pause", {"pid": os.getpid()}, timeout=60)
    print(f"router pause: {'ok' if s == 200 else 'not reached (router off?)'}")

    cmd = ["cmd", "/c", os.path.join(SRC, "bin", os.environ.get("FT_LAUNCHER", "ft.cmd")), "serve", "--model-path", a.model, "--served-model-name", "probe",
           "--host", "127.0.0.1", "--port", str(a.port)] + args
    print(f"[{a.name}] env {extra_env or '-'}  args {' '.join(args)}")
    t0 = time.time(); logf = open(logp, "wb")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, creationflags=0x200)  # new process group
    res = {"name": a.name, "when": time.strftime("%Y-%m-%d %H:%M"), "env": extra_env, "base_env": base_env, "args": args, "model": os.path.basename(a.model), "tests": {}}
    try:
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"server exited ({p.returncode}) during load; see {logp}")
            if "ready to serve" in log_tail(logp, 0)[0]:  # /health answers 200 long before the model is up
                break
            if time.time() - t0 > a.load_timeout:
                raise RuntimeError(f"not ready after {a.load_timeout}s")
            print(f"\r  loading {time.time() - t0:5.0f}s", end="", flush=True); time.sleep(3)
        res["load_s"] = round(time.time() - t0)
        text, pos = log_tail(logp, 0)
        res["load_lines"] = [re.sub(r"\x1b\[[0-9;]*m|^\[[^\]]*\]\s*INFO\s*", "", l).strip()[:240] for l in text.splitlines() if LOAD_RX.search(l)][:25]
        print(f"\r  ready in {res['load_s']}s")
        for l in res["load_lines"]:
            print("   ", l)
        # 2026-11-14 doc 27 A/B probes (graph.py). Undo: *.bak-20261114-probes
        plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
        up = re.findall(r"\[graph-upload\] (hipGraphUpload ok|.*)", plain)
        res["probes"] = {"alloc": [l.strip()[:200] for l in re.findall(r"\[alloc-probe\][^\r\n]*", plain)],
                         "upload_ok": sum(u == "hipGraphUpload ok" for u in up),
                         "upload_other": [u.strip()[:200] for u in up if u != "hipGraphUpload ok"][:3]}
        print(f"    probes: {res['probes']}")
        tests = [t for t in a.tests.split(",") if t]
        plan = {"decode": [(ESSAY, 512, "warm"), (ESSAY, 512, "run")]}
        for t in tests:  # pNk / pN: an N-thousand (or N) token prompt, then 256 tokens of decode at that depth
            m = re.fullmatch(r"p(\d+)(k?)", t)
            if m:
                plan[t] = [(int(m[1]) * (1000 if m[2] else 1), 256, "run")]  # token target; fit_prompt builds it once the server is up
        conc = {t: int(m[1]) for t in tests if (m := re.fullmatch(r"c(\d+)", t)) and int(m[1]) >= 2}
        if bad := [t for t in tests if t not in plan and t not in conc]:
            sys.exit(f"unknown tests: {bad}")
        for i, t in enumerate(tests, 1):
            if t in conc:
                _, pos = log_tail(logp, pos)
                print(f"  [{i}/{len(tests)}] {t} x{conc[t]} concurrent ...", end="", flush=True)
                r = run_concurrent(a.port, conc[t], 512)
                text, pos = log_tail(logp, pos)
                r["stamps"] = stamps_of(text)[-3:]
                res["tests"][t] = r
                print(f" total decode {r['decode_tps']} tok/s ({r['gen_tokens']} tok in {r['wall_s']}s), per-request {r['per_req_tps']}")
                if r["errors"]:
                    raise RuntimeError(f"{t}: {len(r['errors'])} request(s) failed: {r['errors'][0]}")
                continue
            for prompt, mt, kind in plan[t]:
                _, pos = log_tail(logp, pos)
                if isinstance(prompt, int):
                    prompt = fit_prompt(a.port, prompt)
                print(f"  [{i}/{len(tests)}] {t} {kind} ...", end="", flush=True)
                r = chat(a.port, prompt, mt)
                text, pos = log_tail(logp, pos)
                r["stamps"] = stamps_of(text)[-3:]
                print(f" prompt {r['prompt_tokens']} ttft {r['ttft_s']}s prefill {r['prefill_tps']} tok/s, decode {r['decode_tps']} tok/s ({r['gen_tokens']} tok)")
                if kind == "warm" and t == "decode":
                    res["tests"]["decode_warm"] = r  # first request: first replay of each graph (doc 27 B)
                if kind == "run":
                    res["tests"][t] = r
                    for st in r["stamps"][-1:]:
                        print("     ", st[:400])
    except Exception as e:
        res["error"] = str(e)[:500]
        print(f"\n  ERROR: {e}")
        text, _ = log_tail(logp, 0)
        print("  log tail:\n    " + "\n    ".join(text.splitlines()[-12:]))
    finally:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        logf.close()
        with open(os.path.join(OUT, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        print(f"  saved to {os.path.join(OUT, 'results.jsonl')}; router resumes when this exits")


if __name__ == "__main__":
    main()
