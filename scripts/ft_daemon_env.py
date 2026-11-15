"""The engine env the AI-On daemon serves with, read from its launcher, so probes and fits test the served config.

2026-11-14: ft-speedprobe and ft-fit used a fixed pin budget 14 and graph bs 1 while the daemon serves pin 2 + pageable
banks, so probe numbers were not the served config. Rule: the daemon's launcher sets the defaults, the caller's own
environment and --env override them. Left out: the guard's keys, the binary path, and FREETOKEN_HOST_RESERVE_GB (its RAM
preflight opens a window that waits 120 s, wrong for an unattended run). FT_DAEMON_CMD points at another launcher.
Undo: restore ft-speedprobe.py / ft-fit.py from *.bak-20261114-benchsafe and delete this file.
"""
import os
import re

DAEMON_CMD = os.environ.get("FT_DAEMON_CMD") or r"T:\robot\toggle\FT-Daemon-021.cmd"
FALLBACK = {"FREETOKEN_PIN_BUDGET_GB": "14", "FREETOKEN_BANK_CUDA_ALLOC": "1", "FT_GRAPH_MAX_BS": "1"}  # the old fixed set
SKIP = {"FREETOKEN_FT_BIN", "FREETOKEN_HOST_RESERVE_GB"}


def daemon_env(path=DAEMON_CMD):
    """{key: value} of the launcher's `set "FREETOKEN_*=..."` and `set "FT_GRAPH_MAX_BS=..."` lines; FALLBACK if unreadable."""
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return dict(FALLBACK)
    env = {}
    for k, v in re.findall(r'(?im)^\s*set\s+"((?:FREETOKEN_\w+|FT_GRAPH_MAX_BS))=([^"]*)"', text):
        if k.upper() not in SKIP:
            env[k.upper()] = v
    return env or dict(FALLBACK)


if __name__ == "__main__":
    print(DAEMON_CMD)
    for k, v in daemon_env().items():
        print(f"  {k}={v}")
