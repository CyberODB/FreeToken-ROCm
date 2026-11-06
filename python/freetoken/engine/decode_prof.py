"""FREETOKEN_DECODE_PROFILE=N (measurement, ROCm fork 2026-11-04): profile N decode steps once and log
a compact op table to the serve log. Off by default.

The first FREETOKEN_DECODE_PROFILE_SKIP decode steps (default 24) are skipped so Triton JIT/autotune
is not counted. Only the profiled steps sync the GPU (for wall time); every other step runs unchanged.
A profiler error logs a warning and serving carries on."""

from __future__ import annotations

import os
import time

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_STEPS = int(os.getenv("FREETOKEN_DECODE_PROFILE", "0") or 0)
_SKIP = int(os.getenv("FREETOKEN_DECODE_PROFILE_SKIP", "24") or 24)
# "split" (default): CPU enqueue vs GPU finish per step, no profiler. "torch": torch.profiler table
# (on ROCm Windows its GPU times land on view ops and it slows a step ~7x, so only op counts are usable).
_MODE = os.getenv("FREETOKEN_DECODE_PROFILE_MODE", "split").strip().lower()


class DecodeProfiler:
    def __init__(self) -> None:
        self.active = _STEPS > 0
        self.seen = 0
        self.prof = None
        self.walls: list[float] = []

    def _stop(self, why: str) -> None:
        self.active = False
        logger.warning(f"[decode-profile] off: {why}")

    def step(self, batch, fn):
        if not self.active or not batch.is_decode:
            return fn()
        self.seen += 1
        if self.seen <= _SKIP:
            return fn()
        if _MODE != "torch":
            return self._split_step(fn)
        try:
            if self.prof is None:
                from torch.profiler import ProfilerActivity, profile

                self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                self.prof.__enter__()
                logger.info_rank0(f"[decode-profile] recording {_STEPS} decode steps (batch size {batch.size})")
            torch.cuda.synchronize()
        except Exception as exc:
            self._stop(f"start failed: {exc!r}")
            return fn()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        self.walls.append((time.perf_counter() - t0) * 1e3)
        if len(self.walls) >= _STEPS:
            self.active = False
            try:
                self._report()
            except Exception as exc:
                logger.warning(f"[decode-profile] report failed: {exc!r}")
        return out

    def _split_step(self, fn):
        """No profiler: time how long the CPU takes to enqueue the step vs when the GPU finishes it.

        enqueue ~= total means the Python/launch side is the bottleneck (GPU waits on the CPU);
        enqueue << total means the GPU (or the CPU MoE it waits on) is."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.walls.append(((t1 - t0) * 1e3, (t2 - t0) * 1e3))
        if len(self.walls) >= _STEPS:
            self.active = False
            enq = sorted(e for e, _ in self.walls)
            tot = sorted(t for _, t in self.walls)
            med = lambda xs: xs[len(xs) // 2]
            logger.info_rank0(
                f"[decode-profile] split {len(self.walls)} steps: CPU enqueue median {med(enq):.1f} ms "
                f"(min {enq[0]:.1f}, max {enq[-1]:.1f}); GPU done median {med(tot):.1f} ms "
                f"(min {tot[0]:.1f}, max {tot[-1]:.1f}); enqueue/total {100 * med(enq) / max(med(tot), 1e-9):.0f}%"
            )
        return out

    def _report(self) -> None:
        self.prof.__exit__(None, None, None)
        events = self.prof.key_averages()
        n = len(self.walls)
        dev = "self_device_time_total" if events and hasattr(events[0], "self_device_time_total") else "self_cuda_time_total"
        gpu_us = sum(getattr(e, dev, 0) or 0 for e in events)
        cpu_us = sum(e.self_cpu_time_total for e in events)
        launches = sum(e.count for e in events if "launch" in e.key.lower() and "kernel" in e.key.lower())
        kernels = sum(e.count for e in events if (getattr(e, dev, 0) or 0) > 0)
        wall = sum(self.walls) / n
        log = logger.info_rank0
        log(
            f"[decode-profile] {n} steps: wall {wall:.1f} ms/step "
            f"(each {', '.join(f'{w:.1f}' for w in self.walls)}); GPU busy {gpu_us / n / 1e3:.1f} ms/step "
            f"= {100 * gpu_us / 1e3 / max(wall * n, 1e-9):.0f}% of wall; CPU self {cpu_us / n / 1e3:.1f} ms/step "
            f"(profiler-inflated); kernel launch calls {launches / n:.0f}/step; device ops {kernels / n:.0f}/step"
        )
        for title, key, rows in (("by GPU time", dev, 15), ("by CPU time", "self_cpu_time_total", 10)):
            log(f"[decode-profile] top {rows} {title}:")
            for line in events.table(sort_by=key, row_limit=rows, max_name_column_width=60).splitlines():
                log(f"[decode-profile] {line}")


decode_profiler = DecodeProfiler()
