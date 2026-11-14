"""ft-arch-watch: what is coming before a model needs it (2026-11-14).

New model formats show up in llama.cpp's gguf-py (architecture names, tensor types) before anyone ships a GGUF of them, and
FreeToken's own copy of that file (python/freetoken/third_party/gguf, pinned in its VERSION) lags upstream. This compares three
lists and says what changed since the last run:

  engine   GGUF_ARCH_TO_REGISTRY in python/freetoken/models/gguf/config.py: what FreeToken can load
  bundled  MODEL_ARCH_NAMES / GGMLQuantizationType in the bundled constants.py: what its llama.cpp pin knows
  upstream the same file on llama.cpp master (one public GET to raw.githubusercontent.com; --offline skips it)

The watch list is the part to read: architectures upstream or bundled that the engine cannot load, with a version past the newest
it runs in that family (lfm2moe runs, so an lfm3 lands there; gemma3 does not, gemma4 runs). New tensor types upstream are listed too: a quant made with one will not load until
the bundled table and a kernel have it. ft-fit's pre-flight stops on both at load time; this says it weeks earlier.

  py scripts/ft-arch-watch.py            # report, remember what was seen (speedprobe/arch-watch.json)
  py scripts/ft-arch-watch.py --offline  # engine vs bundled only
  py scripts/ft-arch-watch.py --json     # the lists, for other tools
Standard library only. Writes only speedprobe/arch-watch.json.
"""
import argparse, datetime, json, os, re, sys, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, "python", "freetoken")
CONFIG = os.path.join(PY, "models", "gguf", "config.py")
BUNDLED = os.path.join(PY, "third_party", "gguf", "constants.py")
VERSION = os.path.join(PY, "third_party", "gguf", "VERSION")
UPSTREAM = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/gguf-py/gguf/constants.py"
SEEN = os.path.join(REPO, "speedprobe", "arch-watch.json")


def engine_archs(src: str) -> set[str]:
    m = re.search(r"GGUF_ARCH_TO_REGISTRY[^=]*=\s*\{(.*?)\n\}", src, re.S)
    return set(re.findall(r'^\s*"([^"]+)"\s*:', m.group(1), re.M)) if m else set()


def arch_names(src: str) -> set[str]:
    m = re.search(r"^MODEL_ARCH_NAMES[^=]*=\s*\{(.*?)\n\}", src, re.S | re.M)
    return set(re.findall(r'MODEL_ARCH\.\w+\s*:\s*"([^"]+)"', m.group(1))) if m else set()


def quant_types(src: str) -> dict[int, str]:
    m = re.search(r"class GGMLQuantizationType\(IntEnum\):\n(.*?)\n(?=\S)", src, re.S)
    return {int(v): k for k, v in re.findall(r"^\s+([A-Z0-9_]+)\s*=\s*(\d+)", m.group(1), re.M)} if m else {}


def family(name: str) -> str:
    """The letters before the first digit or separator: lfm2moe -> lfm, qwen35moe -> qwen, gemma4 -> gemma."""
    m = re.match(r"[a-z]+", name.lower())
    return m.group(0) if m else name


def version(name: str):
    """The first number after the family letters, read as a version: lfm2 -> 2, qwen35 -> 3.5, minimax-m2 -> 2,
    minimax-01 -> 0.1, glm-dsa -> None (no number)."""
    m = re.search(r"\d+", name.lower()[len(family(name)):])
    if not m:
        return None
    d = m.group(0)
    return float(d[0] + "." + d[1:]) if len(d) > 1 else float(d)


def newer(cands: set[str], eng: set[str], fresh: set[str] = frozenset()) -> list[str]:
    """Names not loadable here whose version is past the newest the engine runs in that family (lfm3 when lfm2 runs),
    plus any name in `fresh` (upstream only) in a family the engine runs. A family whose engine names carry no number
    (gpt-oss, llama) has nothing to order by, so only its fresh names count."""
    top = {}
    for x in eng:
        v = version(x)
        top.setdefault(family(x), None)
        if v is not None:
            top[family(x)] = max(top[family(x)] or 0.0, v)
    out = set()
    for x in cands - eng:
        f, v = family(x), version(x)
        if f in top and (x in fresh or (top[f] is not None and v is not None and v > top[f])):
            out.add(x)
    return sorted(out)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ft-arch-watch"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offline", action="store_true", help="skip the upstream fetch")
    ap.add_argument("--json", action="store_true", help="print the lists as JSON")
    ap.add_argument("--no-save", action="store_true", help="do not update speedprobe/arch-watch.json")
    a = ap.parse_args()

    eng = engine_archs(open(CONFIG, encoding="utf-8").read())
    bsrc = open(BUNDLED, encoding="utf-8").read()
    bun, btypes = arch_names(bsrc), quant_types(bsrc)
    pin = open(VERSION, encoding="utf-8").readline().strip() if os.path.exists(VERSION) else "?"
    if not eng or not bun or not btypes:
        sys.exit(f"could not read the lists (engine {len(eng)}, bundled archs {len(bun)}, types {len(btypes)}): a file layout changed")
    up = utypes = None
    if not a.offline:
        try:
            usrc = fetch(UPSTREAM)
            up, utypes = arch_names(usrc), quant_types(usrc)
            if not up or not utypes:
                print("upstream constants.py read, but its lists were not found (layout changed?); upstream skipped")
                up = utypes = None
        except Exception as e:
            print(f"upstream not reached ({str(e)[:80]}); engine vs bundled only")
    try:
        seen = json.load(open(SEEN, encoding="utf-8"))
    except (OSError, ValueError):
        seen = {}

    fams = {family(x) for x in eng}
    known = up if up is not None else bun
    watch = newer(known, eng, (up - bun) if up is not None else set())
    others = sorted(x for x in known - eng if family(x) in fams and x not in watch)
    up_new = sorted(up - bun) if up is not None else []
    types_new = {k: v for k, v in sorted((utypes or {}).items()) if k not in btypes}
    since = sorted(up - set(seen.get("upstream_archs") or [])) if up is not None and seen.get("upstream_archs") else []
    types_since = sorted(set(map(int, (utypes or {}))) - set(seen.get("upstream_types") or [])) if utypes and seen.get("upstream_types") else []
    out = {"engine": sorted(eng), "bundled_pin": pin, "bundled_archs": len(bun), "upstream_archs": len(up) if up is not None else None,
           "watch": watch, "other_variants": others, "upstream_not_bundled": up_new, "upstream_new_types": types_new, "new_since_last": since,
           "types_new_since_last": [f"{(utypes or {})[k]} ({k})" for k in types_since], "last_run": seen.get("when")}
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        print(f"engine loads {len(eng)} GGUF architectures; bundled gguf: {pin}")
        print(f"bundled knows {len(bun)} architectures, {len(btypes)} tensor types" +
              (f"; llama.cpp master knows {len(up)} and {len(utypes)}" if up is not None else ""))
        print(f"\n1. Watch list ({len(watch)}): in llama.cpp, not loadable here, a newer version than the engine runs in its family")
        for x in watch:
            print(f"   - {x}" + ("  (upstream only: newer than the bundled pin)" if up is not None and x not in bun else ""))
        print(f"   (also not loadable, older or side variants of families it runs: {len(others)})")
        if up is not None:
            print(f"\n2. On llama.cpp master, not in the bundled pin ({len(up_new)}): {', '.join(up_new) or 'none'}")
            print(f"3. Tensor types on master the bundled table lacks ({len(types_new)}): "
                  + (", ".join(f"{v} ({k})" for k, v in types_new.items()) or "none") + ("  -> a quant using one cannot load here yet" if types_new else ""))
            if seen.get("upstream_archs"):
                print(f"4. New on master since the last run ({seen.get('when')}): " + (", ".join(since) or "no architectures")
                      + ("; types " + ", ".join(out["types_new_since_last"]) if types_since else ""))
            else:
                print("4. First run: nothing to compare against yet; the next run lists what is new since today")
    if up is not None and not a.no_save:
        os.makedirs(os.path.dirname(SEEN), exist_ok=True)
        with open(SEEN, "w", encoding="utf-8") as f:
            json.dump({"when": datetime.date.today().isoformat(), "upstream_archs": sorted(up), "upstream_types": sorted(utypes)}, f, indent=1)


if __name__ == "__main__":
    main()
