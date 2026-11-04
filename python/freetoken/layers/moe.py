import os
from typing import TYPE_CHECKING, Iterator, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.moe import is_offload_moe_strategy
from freetoken.moe.fused import fused_topk
from freetoken.moe.offload_cache import OffloadMoeCache


from .base import BaseOP
from .quantization import ExpertView, LayerKind, QuantConfig, quant_method_for

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Router decision (topk_weights[float32], topk_ids[int32]) for models whose router
# is computed outside the MoE layer. Such models call ``routed_forward`` with a
# precomputed routing instead of going through the generic softmax+top-k path.
TopK = Tuple[torch.Tensor, torch.Tensor]

# Hybrid decode overlaps the CPU overflow GEMV behind the GPU PCIe fetch + GEMM by
# default. Set FREETOKEN_HYBRID_OVERLAP=0 to force the serial path (CPU sync before the
# GPU work) -- a measurement-only escape hatch to A/B the overlap benefit.
_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"

# FT_DEBUG_GLU=1 (diagnostic, 2026-11-03): sync after each decode stage of the first decode
# layers and print the stage first, so the last line before a GPU fault names the stage.
_DEBUG_SYNC = bool(os.environ.get("FT_DEBUG_GLU"))
_debug_sync_left = 400

# FREETOKEN_HYBRID_TIMING=N (measurement, 2026-11-03): sync the GPU around each decode MoE phase
# and log ms/step per phase every N decode steps. Serializes the GPU, so compare shares, not
# speed; pair with FREETOKEN_HYBRID_OVERLAP=0 so the CPU work lands in its own bucket. Off by default.
_HYBRID_TIMING = int(os.getenv("FREETOKEN_HYBRID_TIMING", "0") or 0)


class _PhaseTimer:
    """Each mark adds the time since the previous mark to a bucket. Steps start at the first MoE layer."""

    _IDLE_S = 2.0  # a longer gap before a step's first layer is idle time or a prefill: dropped

    def __init__(self, every: int) -> None:
        import time

        from freetoken.utils import init_logger

        self._clock = time.perf_counter
        self._log = init_logger(__name__)
        self.every = every
        self.first_layer: int | None = None
        self._reset()
        self.last: float | None = None

    def _reset(self) -> None:
        self.buckets: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.steps = 0

    def active(self) -> bool:
        return not torch.cuda.is_current_stream_capturing()

    def mark(self, bucket: str) -> None:
        torch.cuda.synchronize()
        now = self._clock()
        if self.last is not None:
            self.buckets[bucket] = self.buckets.get(bucket, 0.0) + now - self.last
        self.last = now

    def count(self, key: str, n: int) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def layer_start(self, layer_id: int, bs: int) -> None:
        if self.first_layer is None:
            self.first_layer = layer_id
        if layer_id != self.first_layer:
            self.mark("between MoE layers (attention, norms, router)")
            return
        torch.cuda.synchronize()
        if self.last is not None and self._clock() - self.last > self._IDLE_S:
            self.last = None  # idle or prefill since the last step
        self.mark("outside MoE layers (step tail, sampling, scheduler)")
        if self.steps >= self.every:
            self.flush()
        self.steps += 1
        self.count(f"steps at bs={bs}", 1)

    def flush(self) -> None:
        n = max(self.steps, 1)
        total = sum(self.buckets.values())
        rows = sorted(self.buckets.items(), key=lambda kv: -kv[1])
        parts = "; ".join(f"{k}: {1e3 * v / n:.1f} ms ({100 * v / total:.0f}%)" for k, v in rows) if total else "-"
        counts = ", ".join(f"{k}={v / n:.1f}" if not k.startswith("steps") else f"{k}={v}" for k, v in self.counts.items())
        self._log.info(f"[hybrid-timing] {n} steps, {1e3 * total / n:.1f} ms/step: {parts} | per step: {counts}")
        self._reset()


_timer = _PhaseTimer(_HYBRID_TIMING) if _HYBRID_TIMING > 0 else None


def _debug_sync(stage: str, layer_id: int) -> None:
    global _debug_sync_left
    if not _DEBUG_SYNC or _debug_sync_left <= 0 or torch.cuda.is_current_stream_capturing():
        return
    _debug_sync_left -= 1
    import time
    t0 = time.perf_counter()
    print(f"[sync-debug] {time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d} layer={layer_id} stage={stage}", flush=True)
    try:
        torch.cuda.synchronize()
    except Exception:
        print(f"[sync-debug] GPU FAULT surfaced at layer={layer_id} stage={stage} after {time.perf_counter() - t0:.3f}s", flush=True)
        raise
    print(f"[sync-debug]   sync took {time.perf_counter() - t0:.3f}s", flush=True)


class MoELayer(BaseOP):
    """Resident routed experts.

    The expert format comes from ``quant_method`` (declared by ``create_weights``, run by
    ``apply``); without a ``quant_config`` the experts are plain bf16. The gated activation is
    ``act(clamp(g, limit) * alpha) * (clamp(u) + beta)`` with ``interleaved`` gate|up rows
    for gpt-oss."""

    quant_layer_kind = LayerKind.MOE

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        allocate_experts: bool = True,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        layer_id: int | None = None,
        strategy: str = "resident",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_rank = tp_info.rank
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.alpha = alpha
        self.beta = beta
        self.limit = limit
        self.interleaved = interleaved
        self.has_bias = has_bias
        self.layer_id = layer_id
        self.strategy = strategy
        self.decode_target = decode_target
        self.prefix = prefix
        self.quant_method = quant_method_for(quant_config, self, prefix)
        if allocate_experts:
            self.quant_method.create_weights(self)

    def finalize(self) -> None:
        self.quant_method.finalize(self)

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            return self._comm.all_reduce(hidden_states)
        return hidden_states

    def _resident_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.quant_method.apply(
            hidden_states, topk_weights, topk_ids, self.quant_method.resident_view(self),
            layer=self, is_prefill=get_global_ctx().batch.is_prefill,
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        Same name and shape as ``OffloadMoELayer.routed_forward`` so a model with
        its own router calls ``experts.routed_forward(...)`` without knowing whether
        the experts are resident or offloaded. The shared contract is the offload
        one: ``topk_ids`` must be safe to mutate in place (the offload decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        The resident path does not mutate it today, but callers must not rely on
        that.

        ``hidden_states`` may also be overwritten by the expert kernel. Compute
        shared branches that need the original input before calling this method.
        """
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._maybe_all_reduce(self._resident_gemm(hidden_states, topk_weights, topk_ids))


class OffloadMoELayer(MoELayer):
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        strategy: str = "offload",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            allocate_experts=False,
            alpha=alpha,
            beta=beta,
            limit=limit,
            interleaved=interleaved,
            has_bias=has_bias,
            layer_id=layer_id,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.offload_cache: OffloadMoeCache | None = None

    @property
    def skips_inactive_slots(self) -> bool:
        return self.quant_method.kernel.supports_inactive_slots

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        ctx = get_global_ctx()
        if ctx.batch.is_prefill:
            final_hidden_states = self.prefill_forward(hidden_states, router_logits)
        else:
            final_hidden_states = self.decode_forward(hidden_states, router_logits)
        return self._maybe_all_reduce(final_hidden_states)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        The entry point for models whose router does not fit ``fused_topk`` (sigmoid
        scores, selection bias, group-limited top-k, ...); identical to ``forward``
        past the router. ``topk_ids`` must be safe to mutate in place (decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.

        ``hidden_states`` may also be overwritten by the expert kernel. Compute
        shared branches that need the original input before calling this method.
        """
        ctx = get_global_ctx()
        if ctx.batch.is_prefill:
            out = self._prefill_routed(hidden_states, topk_weights, topk_ids)
        else:
            out = self._decode_routed(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def decode_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        _debug_sync("decode_start(attn+router done)", self.layer_id)
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        _debug_sync("after_fused_topk", self.layer_id)
        return self._decode_routed(hidden_states, topk_weights, topk_ids)

    def prefill_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._prefill_routed(hidden_states, topk_weights, topk_ids)

    # ------------------------------------------------------------------
    # Data movement -- one decision tree for every quant format (the banks
    # registry makes the cache machinery bank-count agnostic). Decode loads
    # on demand; prefill streams whole layers, double-buffered when overlap
    # is enabled. The kernels only ever see bank views plus row indices;
    # which kernel runs is decided afterwards, in ``_expert_gemm``.
    # ------------------------------------------------------------------

    def _decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """On-demand load: ``ensure_experts`` rewrites ``topk_ids`` into cache slot
        ids in place (loading missing experts), then the GEMM reads the full slot
        cache. All device-side with fixed shapes, so the decode call is CUDA-graph
        capturable.

        For ``decode_target == "cpu"`` the experts are instead computed on the CPU
        (high RAM bandwidth) straight from the host banks: ship hidden/routing to
        pinned host memory, run the GEMV on the worker pool via host nodes, ship the
        result back. The GPU slot cache is untouched (topk_ids keep their raw expert
        ids), so no ``ensure_experts``/``copy_missing`` here."""
        cache = self.offload_cache
        assert cache is not None
        timer = _timer if _timer is not None and _timer.active() else None
        if timer is not None:
            timer.layer_start(self.layer_id, hidden_states.shape[0])
        if cache.is_cpu_layer(self.layer_id):
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            out = executor.decode(self.layer_id, hidden_states, topk_weights, topk_ids)
            if timer is not None:
                timer.mark("CPU-only layers: CPU MoE")
                timer.count("CPU-only layers", 1)
                timer.count("CPU-only layer routes", int((topk_ids >= 0).sum()))
            return out
        if cache.decode_target == "hybrid":
            return self._decode_hybrid(cache, hidden_states, topk_weights, topk_ids, timer)
        if timer is not None:
            timer.count("plain GPU layers", 1)
        cache.ensure_experts(self.layer_id, topk_ids)
        if timer is not None:
            timer.mark("plain layers: ensure_experts")
        cache.copy_missing()
        if timer is not None:
            timer.mark("plain layers: PCIe fetch")
        out = self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.pool_bank_views(self.layer_id),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        if timer is not None:
            timer.mark("plain layers: GPU expert GEMM")
        return out

    def _decode_hybrid(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        timer: "_PhaseTimer | None" = None,
    ) -> torch.Tensor:
        """Hybrid decode: GPU computes cache hits + <=K freshly-fetched experts, the CPU
        computes the overflow misses, overlapped, then the partials merge.

        The CPU pool is kicked off (``decode_submit``) before the GPU PCIe fetch + GEMM so
        the CPU overflow GEMV runs concurrently with the GPU work. Capture-safe: the
        routing split runs inside the hybrid ensure kernel and the CPU submit/sync are host nodes.
        Each route is computed exactly once -- CPU-assigned routes reach the GPU as slot -1 with
        weight 0 (a kernel with ``supports_inactive_slots`` skips them without a read; any other reads
        slot 0, whose storage is a payload or the bank's zero fill) and the CPU ids are -1 for
        GPU-assigned routes (the C++ kernel skips id<0).
        """
        executor = cache.cpu_executor
        assert executor is not None, "CPU MoE executor was not initialized"
        # the kernel reads every route's slot: give the inactive ones slot 0, which holds a
        # complete payload or the bank's zero fill (never uninitialized storage), times weight 0
        # with several quants this holds only because the cache zeroes its staging area after prefill
        clamp = not self.skips_inactive_slots
        gpu_slots, gpu_w, cpu_ids = cache.ensure_experts_hybrid(self.layer_id, topk_ids, topk_weights, clamp=clamp)
        _debug_sync("after_ensure_experts_hybrid", self.layer_id)
        kind = "locked" if self.layer_id in cache.locked_hybrid else "pinned"
        if timer is not None:
            timer.mark(f"hybrid {kind}: ensure/routing split")
            timer.count(f"hybrid {kind} layers", 1)
            timer.count(f"hybrid {kind} CPU routes", int((cpu_ids >= 0).sum()))
            timer.count(f"hybrid {kind} GPU routes", int((gpu_w != 0).sum()))
        pending = executor.decode_submit(self.layer_id, hidden_states, topk_weights, cpu_ids)
        _debug_sync("after_cpu_decode_submit", self.layer_id)
        if timer is not None:
            timer.mark(f"hybrid {kind}: CPU submit")

        # Measurement knob: FREETOKEN_HYBRID_OVERLAP=0 syncs the CPU pool *before* the
        # PCIe fetch + GPU GEMM, serializing the two so an A/B isolates the overlap win.
        # FT_DEBUG_GLU: while the debug checkpoints run, finish the CPU work first so a fault can be pinned on it
        overlap = _HYBRID_OVERLAP and not (_DEBUG_SYNC and _debug_sync_left > 0)
        cpu_routed_early = (
            executor.decode_sync(pending) if not overlap else None
        )
        if not overlap:
            _debug_sync("after_cpu_decode_sync_early", self.layer_id)
            if timer is not None:
                timer.mark(f"hybrid {kind}: CPU MoE (serial)")

        cache.copy_missing()
        _debug_sync("after_copy_missing", self.layer_id)
        if timer is not None:
            timer.mark(f"hybrid {kind}: PCIe fetch")
        gpu_routed = self._expert_gemm(
            cache,
            hidden_states,
            gpu_w,
            gpu_slots,
            views=cache.pool_bank_views(self.layer_id),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        _debug_sync("after_gpu_expert_gemm", self.layer_id)
        if timer is not None:
            timer.mark(f"hybrid {kind}: GPU expert GEMM")
        if not overlap:
            return gpu_routed + cpu_routed_early
        out = executor.decode_sync(pending, add_to=gpu_routed)
        _debug_sync("after_cpu_decode_sync", self.layer_id)
        if timer is not None:
            timer.mark(f"hybrid {kind}: CPU wait (not hidden)")
        return out

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill movement: stream whole layers -- double-buffered behind the
        previous layer's GEMMs when ``prefill_overlap`` is on, else a synchronous
        ``materialize_layer``. In both, position == expert id, so the routing ids
        pass through unmapped."""
        cache = self.offload_cache
        assert cache is not None
        if cache.prefill_overlap:
            views = self._wait_prefill_overlap(cache)
            out = self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=views,
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
            cache.release_prefill_layer(self.layer_id)
            return out
        cache.materialize_layer(self.layer_id)
        cache.copy_missing()
        out = self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.stage_views(self.layer_id, 0),
            n=self.num_experts,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=True,
        )
        cache.release_prefill_layer(self.layer_id)
        return out

    def _wait_prefill_overlap(self, cache: OffloadMoeCache) -> tuple[torch.Tensor, ...]:
        """Double-buffer choreography for this layer's overlap prefill: kick off the
        next layer's full-layer H2D copy, then return this layer's bank views (in
        bank registration order; buffer position == expert id, so routing ids pass
        through unmapped). The caller runs ``release_prefill_layer`` after its GEMMs.
        """
        if self.layer_id == 0:
            cache.begin_prefill()
        cache.prefetch_prefill_layer(self.layer_id)
        cache.prefetch_prefill_layer(self.layer_id + 1)
        return cache.wait_prefill_layer(self.layer_id)

    # ------------------------------------------------------------------
    # Kernel dispatch: ``views`` are the bank tensors the movement step produced (in bank registration order) and ``topk_ids`` already index their rows.
    # ------------------------------------------------------------------

    def _expert_gemm(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        views: tuple[torch.Tensor, ...],
        n: int | None,
        alphas: tuple[torch.Tensor, torch.Tensor] | None,
        is_prefill: bool,
    ) -> torch.Tensor:
        from freetoken.moe.legacy_format import canonical_role  # legacy_format imports this package

        view = ExpertView(
            {canonical_role(name): t for name, t in zip(cache.bank_schema, views)},
            slots=None if n is not None else topk_ids, n=n, alphas=alphas,
        )
        return self.quant_method.apply(
            hidden_states, topk_weights, topk_ids, view, layer=self, is_prefill=is_prefill
        )


def iter_moe_layers(model) -> Iterator[MoELayer]:
    if isinstance(model, MoELayer):
        yield model
    if not isinstance(model, BaseOP):
        return
    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_moe_layers(item)


def make_moe_layer(
    config: "ModelConfig",
    *,
    layer_id: int | None = None,
    activation: str = "silu",
    renormalize: bool | None = None,
    apply_router_weight_on_input: bool = False,
    num_experts: int | None = None,
    top_k: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    resident_cls: type[MoELayer] | None = None,
    offload_cls: "type[OffloadMoELayer] | None" = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    limit: float | None = None,
    interleaved: bool = False,
    has_bias: bool = False,
    quant_config: QuantConfig | None = None,
    prefix: str = "",
) -> MoELayer:
    """Build the experts layer for ``config.moe_strategy`` -- the one construction
    seam between a model and the MoE strategy.

    Picks ``OffloadMoELayer`` for the offload family (offload/cpu/hybrid) and
    ``MoELayer`` otherwise. Geometry defaults come from ``config``; pass overrides
    for models whose fields deviate. ``resident_cls``/``offload_cls`` keep model-specific
    subclasses constructible through the same seam.
    """
    offload = is_offload_moe_strategy(config.moe_strategy)
    layer_cls = (offload_cls or OffloadMoELayer) if offload else (resident_cls or MoELayer)
    kwargs = dict(
        num_experts=num_experts if num_experts is not None else config.num_experts,
        top_k=top_k if top_k is not None else config.num_experts_per_tok,
        hidden_size=hidden_size if hidden_size is not None else config.hidden_size,
        intermediate_size=(
            intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        ),
        renormalize=renormalize if renormalize is not None else config.norm_topk_prob,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        alpha=alpha,
        beta=beta,
        limit=limit,
        interleaved=interleaved,
        has_bias=has_bias,
        quant_config=quant_config,
        prefix=prefix,
        layer_id=layer_id,
    )
    if offload:
        assert layer_id is not None, "offload MoE backends need the layer_id"
        kwargs["strategy"] = config.moe_strategy
        kwargs["decode_target"] = config.decode_target
    return layer_cls(**kwargs)
