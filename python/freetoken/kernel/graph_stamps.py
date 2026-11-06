"""In-graph decode timestamps: ``FREETOKEN_GRAPH_STAMPS=N`` logs ms per phase every N decode steps.

A one-thread kernel adds the GPU wall-clock time since the previous stamp to a per-(phase, layer)
slot. Stamps are only recorded while a decode graph is being captured (plus one after sampling,
outside the graph, for graph-replayed steps), so eager prefill and warmup add nothing, and with the
variable unset no stamp is captured at all. ``1`` means every 64 steps.

The time between the after-sampling stamp and the next graph's first stamp is the host gap: GPU time
spent outside the graph (input copies, scheduler, token readback), or idle. Gaps over 200 ms
(between requests) are dropped.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import torch

from freetoken.utils import init_logger

from .utils import load_jit

logger = init_logger(__name__)

ENV = "FREETOKEN_GRAPH_STAMPS"
MAX_SLOTS = 1024
MAX_GAP_MS = 200


def _every() -> int:
    try:
        n = int(os.environ.get(ENV, "0") or 0)
    except ValueError:
        return 0
    return 64 if n == 1 else max(n, 0)


EVERY = _every()


@lru_cache(maxsize=None)
def _module():
    return load_jit(
        "graph_stamps",
        cuda_files=["graph_stamps.cuh"],
        cuda_wrappers=[("stamp", "graph_stamp"), ("ticks_per_ms", "graph_stamp_ticks_per_ms")],
    )


class GraphStamps:
    def __init__(self, every: int, device: torch.device):
        self.every = every
        self.device = device
        self.mod = _module()
        self.ticks_per_ms = int(self.mod.ticks_per_ms(device.index or 0))
        if self.ticks_per_ms <= 0:
            raise RuntimeError("graph stamps: wall clock rate unavailable")
        self.max_delta = self.ticks_per_ms * MAX_GAP_MS
        self.state = torch.zeros(1 + 2 * MAX_SLOTS, dtype=torch.int64, device=device)
        self.slots: Dict[Tuple[str, int], int] = {}
        self.order: List[str] = []
        self.steps = 0
        self.counter_fn: Callable[[], torch.Tensor | None] | None = None
        self._last_counters: torch.Tensor | None = None
        logger.info(f"[graph-stamps] on: log every {every} decode steps, clock {self.ticks_per_ms} ticks/ms")

    def _slot(self, phase: str, layer: int) -> int:
        key = (phase, layer)
        s = self.slots.get(key)
        if s is None:
            if len(self.slots) >= MAX_SLOTS:
                return -1
            s = self.slots[key] = len(self.slots)
            if phase not in self.order:
                self.order.append(phase)
        return s

    def _launch(self, slot: int) -> None:
        self.mod.stamp(self.state, MAX_SLOTS, slot, self.max_delta)

    def stamp(self, phase: str, layer: int = -1) -> None:
        """Time since the previous stamp goes to ``phase``. Recorded only during graph capture."""
        if torch.cuda.is_current_stream_capturing():
            self._launch(self._slot(phase, layer))

    def after_sample(self) -> None:
        """Called after sampling a graph-replayed step: stamps sampling, logs every N steps."""
        self._launch(self._slot("sampling (outside graph)", -1))
        self.steps += 1
        if self.steps % self.every == 0:
            self._report()

    def _report(self) -> None:
        st = self.state.cpu()
        self.state[1:].zero_()
        acc = st[1 : 1 + MAX_SLOTS].tolist()
        cnt = st[1 + MAX_SLOTS :].tolist()
        n = self.every
        by_phase: Dict[str, float] = {}
        per_layer: Dict[str, Dict[int, float]] = {}
        for (phase, layer), s in self.slots.items():
            ms = acc[s] / self.ticks_per_ms / n
            by_phase[phase] = by_phase.get(phase, 0.0) + ms
            if layer >= 0:
                per_layer.setdefault(phase, {})[layer] = ms
        total = sum(by_phase.values())
        parts = ", ".join(f"{p} {by_phase[p]:.2f}" for p in self.order if by_phase.get(p, 0) > 0)
        gap_steps = cnt[self.slots[("host gap", -1)]] if ("host gap", -1) in self.slots else 0
        logger.info(
            f"[graph-stamps] {n} steps: {total:.2f} ms/step = {parts} "
            f"(ms/step; host gap seen on {gap_steps}/{n} steps)"
        )
        pre = per_layer.get("rest of layer (norm, attn/GDN, shared, router)")
        if pre:
            logger.info(
                "[graph-stamps] rest-of-layer us by layer: "
                + " ".join(f"{l}:{pre[l] * 1000:.0f}" for l in sorted(pre))
            )
        self._report_counters(n)

    def _report_counters(self, n: int) -> None:
        if self.counter_fn is None:
            return
        cur = self.counter_fn()
        if cur is None:
            return
        cur = cur.cpu()
        prev, self._last_counters = self._last_counters, cur
        if prev is None:
            return
        d = (cur - prev).tolist()  # rows: layer; cols: missing, fetched, active, steps
        tot = [0, 0, 0]
        rows = []
        for layer, (miss, fetched, active, steps) in enumerate(d):
            if steps <= 0 or active <= 0:
                continue
            hits, cpu = active - miss, miss - fetched
            tot[0] += hits
            tot[1] += fetched
            tot[2] += cpu
            rows.append(f"{layer}:{100 * hits / active:.0f}%/{fetched / steps:.1f}f/{cpu / steps:.1f}c")
        routes = sum(tot)
        if routes:
            logger.info(
                f"[graph-stamps] routes per step: hits {tot[0] / n:.1f}, PCIe fetch {tot[1] / n:.1f}, "
                f"CPU {tot[2] / n:.1f} (hit rate {100 * tot[0] / routes:.1f}%)"
            )
            logger.info("[graph-stamps] by layer hit%/fetch/cpu: " + " ".join(rows))


_stamps: GraphStamps | None = None


def get() -> GraphStamps | None:
    """The process-wide stamper, or None when ``FREETOKEN_GRAPH_STAMPS`` is unset."""
    global _stamps
    if EVERY <= 0:
        return None
    if _stamps is None and torch.cuda.is_available():
        _stamps = GraphStamps(EVERY, torch.device("cuda", torch.cuda.current_device()))
    return _stamps


def stamp(phase: str, layer: int = -1) -> None:
    s = get() if EVERY > 0 else None
    if s is not None:
        s.stamp(phase, layer)
