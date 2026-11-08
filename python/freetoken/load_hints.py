"""Plain-language "what to try" for a failed model load (ROCm fork, 2026-11-08).

``explain(message)`` matches the text of a load error against the known ones and returns a short, ordered
nudge ("try lowering context and KV first"), or None when nothing matches. Used by the API server's backend-failure
log line and by scripts/ft-fit.py. Standard library only, so any script can import it by file path.
Order of advice: the cheapest change that keeps quality first (context / KV), then expert cache, then ratio, then closing apps.
"""
from __future__ import annotations

import math
import re

_KV = "lower the context (--max-seq-len-override) or KV (--num-pages, --kv-reserve-tokens) first"


def _gib(m: str) -> float:
    return float(m)


def _pin(msg: str) -> str | None:
    m = re.search(r"expert banks need ([\d.]+) GiB of pinned host RAM but the pin budget is ([\d.]+) GiB", msg)
    if m:
        need, have = _gib(m[1]), _gib(m[2])
        return (f"The expert banks need {need:.1f} GiB pinned but the pin budget is {have:.1f} GiB "
                f"({need - have:.1f} GiB short). Try: set FREETOKEN_PIN_BUDGET_GB={math.ceil(need + 0.5)} if host RAM allows, "
                "or pass --moe-cpu-layers auto to run the extra layers on the CPU.")
    m = re.search(r"banks ([\d.]+) GiB exceed the pin budget ([\d.]+) GiB", msg)
    if m:
        return (f"The CPU MoE path can't serve this model and the banks ({_gib(m[1]):.1f} GiB) are over the pin budget "
                f"({_gib(m[2]):.1f} GiB). Try: FREETOKEN_PIN_BUDGET_GB={math.ceil(_gib(m[1]) + 0.5)}, or a smaller quant.")
    return None


# (regex on the error text, advice). First match wins; a callable gets the message and may return None to fall through.
RULES: list[tuple[str, object]] = [
    (r"expert banks need .* pin budget|banks .* exceed the pin budget", _pin),
    (r"several MoE slot pools.*--moe-cpu-layers",
     "The pin budget is smaller than the expert banks, so layers were locked to the CPU, and this slot-pool setup (per-layer "
     "expert cache) can't combine with that. Try: raise FREETOKEN_PIN_BUDGET_GB above the bank size so nothing is locked, "
     "or pass --moe-locked-fetch 0, or switch --moe-cache-scope to global."),
    (r"cache budget too small|not enough memory for kv cache|Not enough memory for KV cache",
     f"The GPU has no room left for the KV cache after weights and the expert cache. Try, in order: {_KV}; "
     "then a smaller --moe-cache-size; then raise --memory-ratio only if free VRAM allows; "
     "close apps that hold VRAM (browser with hardware acceleration, games)."),
    (r"slot cap \d+ below the minimum",
     "The expert cache is smaller than the minimum this model needs. Try: raise --moe-cache-size (or drop the cap)."),
    (r"MoE cache of \d+ slots exceeds",
     "The expert cache asks for more slots than the pools can hold. Try: lower --moe-cache-size."),
    (r"out of memory|OutOfMemory|hipErrorOutOfMemory|CUDA error: out of memory|failed to allocate|Tried to allocate",
     f"The GPU ran out of VRAM while loading. Try, in order: {_KV}; then a smaller --moe-cache-size; "
     "then lower --memory-ratio by 0.05; close apps that hold VRAM. If the same config fit before, check for a leftover server "
     "still holding VRAM."),
    (r"WinError 1455|paging file is too small|MemoryError|Cannot allocate memory|ENOMEM|not enough (host|system) (ram|memory)",
     "The PC ran out of host RAM or page file while loading. Try: close big apps (browser, Discord, Claude), "
     "enlarge the Windows page file, or serve a smaller quant."),
    # reasons the GPU guard (scripts/gpu-guard.py) gives when it stops a run
    (r"free VRAM [\d.]+ GiB under",
     f"The guard stopped it: the GPU ran out of free memory. Try, in order: {_KV}; then a smaller --moe-cache-size; "
     "then lower --memory-ratio by 0.05."),
    (r"available RAM [\d.]+ GiB under .* while paging in",
     "The guard stopped it: the PC was out of RAM and swapping. Try: close big apps before loading, lower the context or KV, "
     "or serve a smaller quant."),
    (r"available RAM [\d.]+ GiB under",
     "The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant."),
    (r"free commit [\d.]+ GiB under",
     "The guard stopped it: Windows nearly ran out of commit (RAM + page file). Try: close big apps or enlarge the page file."),
    (r"prompt is too long",
     "The prompt is bigger than the loaded context. Try: raise --max-seq-len-override if VRAM allows, or send less."),
    (r"Backend worker is gone|did not become ready",
     "The model process died while loading. The real reason is the first ERROR/Traceback above this line in the serve log."),
]


def explain(message: str) -> str | None:
    """Advice for a load-failure message, or None when it is not a known one."""
    for pat, advice in RULES:
        if re.search(pat, message, re.I | re.S):
            out = advice(message) if callable(advice) else advice
            if out:
                return out
    return None
