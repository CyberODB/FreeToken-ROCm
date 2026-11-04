"""Thin, defensive OS/`/proc` helpers for the supervisor: spawn-group signalling, footprint,
orphan detection. Every function is best-effort: on a non-Linux host (no ``/proc``) or a vanished
pid it returns a safe default instead of raising, so the daemon never goes down over an
environment quirk ("degraded start"). Imports stdlib only."""

from __future__ import annotations

import errno
import os
import signal
import time
from typing import Iterable

_HAS_PROC = os.path.isdir("/proc")


def pid_alive(pid: int) -> bool:
    """True iff a process with this pid currently exists. ``kill(pid, 0)`` is the portable probe:
    ESRCH → gone, EPERM → exists but not ours (still alive)."""
    if pid <= 0:
        return False
    if os.name == "nt":  # patched: os.kill(pid, 0) TERMINATES the process on Windows
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5 or ctypes.windll.kernel32.GetLastError() == 5
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:  # pragma: no cover - defensive
        return exc.errno != errno.ESRCH
    return True


def _read_proc(pid: int, name: str) -> str | None:
    try:
        with open(f"/proc/{pid}/{name}", "rb") as fh:
            return fh.read().decode("utf-8", "replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    except OSError:  # pragma: no cover - defensive
        return None


def read_cmdline(pid: int) -> list[str]:
    """The process argv as a list (``/proc/<pid>/cmdline`` is NUL-separated). [] if unavailable."""
    raw = _read_proc(pid, "cmdline")
    if not raw:
        return []
    parts = raw.split("\x00")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def _stat_fields(pid: int) -> list[str] | None:
    """Split ``/proc/<pid>/stat`` past the comm field (which may itself contain spaces/parens).
    Returns the fields from ``state`` onward, so field N (N>=3) is ``fields[N-3]``."""
    raw = _read_proc(pid, "stat")
    if not raw:
        return None
    rparen = raw.rfind(")")
    if rparen == -1:
        return None
    return raw[rparen + 2 :].split()


def read_starttime(pid: int) -> int | None:
    """The process start time (stat field 22, clock ticks since boot). Combined with the pid this
    is a stable identity that survives nothing — it changes on PID reuse — so it is the guard
    against adopting/signalling a recycled pid."""
    fields = _stat_fields(pid)
    if fields is None or len(fields) < 20:
        return None
    try:
        return int(fields[19])  # field 22 == fields[22-3]
    except (ValueError, IndexError):  # pragma: no cover - defensive
        return None


def proc_pgid(pid: int) -> int | None:
    """The process group id (stat field 5). Used to (a) enumerate the serve's worker tree and
    (b) assert ``pgid == pid`` before a group-kill so we never signal an unrelated group."""
    fields = _stat_fields(pid)
    if fields is None or len(fields) < 3:
        return None
    try:
        return int(fields[2])  # field 5 == fields[5-3]
    except (ValueError, IndexError):  # pragma: no cover - defensive
        return None


def _iter_all_pids() -> Iterable[int]:
    try:
        for name in os.listdir("/proc"):
            if name.isdigit():
                yield int(name)
    except OSError:  # pragma: no cover - defensive
        return


def tree_pids(root_pid: int) -> list[int]:
    """The serve and all its worker processes: every pid whose process group == ``root_pid``.
    The serve is spawned as a session/group leader (``start_new_session=True``), and its
    mp-spawned scheduler/tokenizer workers inherit that group, so pgid membership captures the
    whole tree without walking ppid chains. Falls back to ``[root_pid]`` off-/proc."""
    if not _HAS_PROC:
        if os.name == "nt":  # patched: no pgid on Windows; walk the parent chain instead
            try:
                import psutil

                root = psutil.Process(root_pid)
                return [root_pid] + [c.pid for c in root.children(recursive=True)]
            except Exception:  # noqa: BLE001 -- psutil missing or root gone
                pass
        return [root_pid] if pid_alive(root_pid) else []
    out = []
    for pid in _iter_all_pids():
        if proc_pgid(pid) == root_pid:
            out.append(pid)
    if root_pid not in out and pid_alive(root_pid):
        out.append(root_pid)
    return out


# patched: Windows has no parent-death signal, and os.kill(self, SIGTERM) there is TerminateProcess,
# which skips every Python-level teardown. A job object with KILL_ON_JOB_CLOSE makes the OS kill
# all workers the moment the serve process dies, however it dies. The handle is held for the
# process lifetime: closing it early would kill the serve itself (it is in the job too).
_JOB_HANDLE = None


def kill_workers_with_me() -> bool:
    """Windows only: put this process in a kill-on-close job so every child spawned AFTER this
    call dies with it. Call before the first worker starts. No-op (False) elsewhere or on failure."""
    global _JOB_HANDLE
    if os.name != "nt" or _JOB_HANDLE is not None:
        return _JOB_HANDLE is not None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        class _Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class _Extended(ctypes.Structure):
            _fields_ = [("Basic", _Basic), ("Io", _IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)  # NULL security attrs: not inheritable
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObject")
        info = _Extended()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):  # Extended info
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
        _JOB_HANDLE = job
        return True
    except Exception as exc:  # noqa: BLE001 -- never block startup over this
        import logging

        logging.getLogger(__name__).warning("worker job object not set up (%s); workers may outlive a crash", exc)
        return False


def signal_group(pid: int, sig: int) -> None:
    """Signal the serve's whole process group. Resolves the pgid and asserts ``pgid == pid``
    (the group-leader invariant of our own spawn) before ``killpg``; on any mismatch or lookup
    failure it falls back to signalling just the pid, so a re-adopted process with an unexpected
    group is never able to make us nuke an innocent group."""
    if os.name == "nt":  # patched: kill the whole tree via taskkill
        import subprocess

        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    pgid = proc_pgid(pid)
    try:
        if pgid is not None and pgid == pid:
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except OSError:  # pragma: no cover - defensive
        pass


def set_oom_score_adj(pid: int, score: int) -> bool:
    """Best-effort write to ``/proc/<pid>/oom_score_adj``. Returns whether it stuck.
    Lowering the daemon's own score needs privilege we may not have under ``systemctl --user``;
    raising the serve's is the load-bearing half and usually succeeds. Never raises."""
    try:
        with open(f"/proc/{pid}/oom_score_adj", "w") as fh:
            fh.write(str(score))
        return True
    except OSError:
        return False


def read_pss_bytes(pid: int) -> int:
    """Proportional set size for one process from ``/proc/<pid>/smaps_rollup`` (shared pages
    counted fractionally — the honest per-process RAM number). 0 if unavailable."""
    raw = _read_proc(pid, "smaps_rollup")
    if not raw:
        return 0
    for line in raw.splitlines():
        if line.startswith("Pss:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024  # value is in kB
    return 0


def is_ft_serve_on_port(pid: int, port: int, *, starttime: int | None = None) -> bool:
    """Verify ``pid`` is (still) an ``ft serve`` bound to ``port`` — the re-adoption / liveness
    identity check. Requires: alive, unchanged start time (PID-reuse
    guard), an argv that looks like our serve invocation, and a matching ``--port`` (or the serve
    default when unspecified). Off-/proc it degrades to a bare liveness check."""
    if not pid_alive(pid):
        return False
    if starttime is not None and read_starttime(pid) != starttime:
        return False
    if not _HAS_PROC:
        return True
    argv = read_cmdline(pid)
    if not argv:
        return False
    joined = " ".join(argv)
    looks_like_serve = "serve" in argv and (
        "freetoken.cli" in joined or "freetoken" in joined or os.path.basename(argv[0]) in {"ft", "ft.exe"}
    )
    if not looks_like_serve:
        return False
    return _argv_port(argv) == port


def _argv_port(argv: list[str], default: int = 1919) -> int:
    """The integer following a standalone ``--port`` / ``-p`` in argv, else the serve default
    (``ServerArgs.server_port``). Also accepts ``--port=N``."""
    for i, tok in enumerate(argv):
        if tok in ("--port", "-p") and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.isdigit():
                return int(nxt)
        elif tok.startswith("--port="):
            val = tok.split("=", 1)[1]
            if val.isdigit():
                return int(val)
    return default


def now_monotonic() -> float:
    return time.monotonic()
