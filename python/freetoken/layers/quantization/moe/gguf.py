"""GGUF experts: packed ggml rows in two banks (gate|up and down), multiplied by llama.cpp's mul_mat_id kernels; float experts run on the bf16 fused MoE."""

from __future__ import annotations

import os
import math

import torch

from ..ggml_types import NVFP4, float_dtype, is_quantized, row_bytes, type_name
from ..registry import LayerKind, register_method
from ..scheme import QuantKind, gguf_types
from .base import BankSpec, ExpertView, MoEConfig, MoEKernel, MoEMethod, gated_epilogue_reason, is_resident, limit_or_inf



def fused_glu(layer) -> tuple[str, float] | None:
    """The MMVQ epilogue (``ggml.mul_mat_id_glu``) that computes this layer's gated activation, or None to run it as a separate kernel."""
    act, alpha, limit = layer.activation, float(layer.alpha), limit_or_inf(layer)
    if act == "silu" and alpha == 1.0 and limit == math.inf:
        return "swiglu", 0.0
    if act == "gelu_tanh":
        return "geglu", 0.0
    if act == "swiglu_clamp" and alpha == 1.0 and limit != math.inf:
        return "swiglu_clamp", limit
    # ggml's fused swiglu_oai keeps its default alpha and limit
    if act == "swigluoai" and math.isclose(alpha, 1.702, rel_tol=1e-6) and math.isclose(limit, 7.0, rel_tol=1e-6):
        return "swiglu_oai", 0.0
    return None


def expert_types(cfg: MoEConfig) -> tuple[int, int]:
    gate, up, down = gguf_types(cfg.scheme)
    if gate != up:
        raise NotImplementedError(f"GGUF experts with gate {type_name(gate)} and up {type_name(up)} need one gate_up bank type")
    return gate, down


def _bank_spec(ggml_type: int, rows: int, k: int, bf16: bool) -> BankSpec:
    # a float bank loads and runs bf16 when the whole layer is float (the fused MoE's dtype);
    # the float half of a mixed layer keeps the file's dtype, as on main
    if float_dtype(ggml_type) is None:
        return BankSpec((rows, row_bytes(ggml_type, k)), torch.uint8)
    return BankSpec((rows, k), torch.bfloat16 if bf16 else float_dtype(ggml_type))


_DECODE_CONFIG = {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "num_warps": 8}


def float_mul_mat_id(w: torch.Tensor, x: torch.Tensor, ids: torch.Tensor, topk_weights: torch.Tensor, config: dict | None) -> torch.Tensor:
    """``ggml.mul_mat_id`` for float experts ``w[E, N, K]``: same ``x`` and ``ids``, same ``[T, k, N]`` output in ``x``'s dtype, no router weights.

    ``config`` runs the sorted-route kernel (prefill, resident layers); None runs the per-route decode kernel."""
    from freetoken.kernel import fused_moe_decode_kernel_triton, fused_moe_kernel_triton
    from freetoken.moe.fused import moe_align_block_size

    t, top_k = ids.shape
    per_route = x.dim() == 3
    # the bank takes its own dtype's activations: bf16 for a whole-layer float layer, the file's dtype for a mixed layer's float half
    a = x.reshape(-1, x.shape[-1]).to(w.dtype).contiguous()
    out = torch.empty(t, top_k, w.shape[1], dtype=x.dtype, device=x.device)
    if config is None:
        fused_moe_decode_kernel_triton(a, w, out, topk_weights, ids, False, per_route, _DECODE_CONFIG, compute_type=x.dtype)
    else:
        sorted_ids, expert_ids, num_post_padded = moe_align_block_size(ids, config["BLOCK_SIZE_M"], w.shape[0])
        fused_moe_kernel_triton(
            a, w, out, topk_weights, ids, sorted_ids, expert_ids, num_post_padded, False, 1 if per_route else top_k, config, compute_type=x.dtype
        )
    return out



class GgmlMoEKernel(MoEKernel):
    name = "ggml"
    cpu_format = "gguf"
    supports_inactive_slots = True

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        if not is_quantized(expert_types(cfg)[0]):
            reason = gated_epilogue_reason(cfg)
            if reason:
                return f"float GGUF experts run on the bf16 fused MoE: {reason}"
        if cfg.has_bias and cfg.decode_target != "gpu":
            return "gpt-oss per-expert biases have no CPU executor format; decode must run on the GPU"
        return self._common_reject(cfg, tp_ok=False, cpu_ok=True, plain_silu_only=False, bias_ok=True)

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        gate_up, down = expert_types(cfg)
        i, h = cfg.local_intermediate, cfg.hidden
        bf16 = float_dtype(gate_up) is not None and float_dtype(down) is not None
        banks = {"gate_up": _bank_spec(gate_up, 2 * i, h, bf16), "down": _bank_spec(down, h, i, bf16)}
        if cfg.scheme.weight.scale is not None:
            # each expert's NVFP4 weight_scale_2 as (gate, up, down, 1): a 16-byte row, which the slot cache's copies need
            banks["global"] = BankSpec((4,), torch.float32)
        if cfg.has_bias:
            # gpt-oss per-expert biases: bank rows so they follow the expert through the slot cache (decode)
            # and the expert-ordered prefill / resident views, as llama.cpp's build_moe_ffn applies them
            banks["gate_up_bias"] = BankSpec((2 * i,), torch.bfloat16)
            banks["down_bias"] = BankSpec((h,), torch.bfloat16)
        return banks

    def pack(self, pieces, cfg: MoEConfig, out):
        i = cfg.local_intermediate
        out["gate_up"][:, :i].copy_(pieces["gate"])
        out["gate_up"][:, i:].copy_(pieces["up"])
        out["down"].copy_(pieces["down"])
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.kernel import ggml
        from freetoken.layers.activation import gated_act_and_mul

        cfg = layer.quant_method.cfg
        gate_up_type, down_type = expert_types(cfg)
        if not is_quantized(gate_up_type) and not is_quantized(down_type):
            return self._apply_float(layer, x, topk_weights, topk_ids, view, gate_up_type, is_prefill=is_prefill)
        assert not layer.apply_router_weight_on_input, "GGUF experts apply the router weights on the output"
        ids = topk_ids.to(torch.int32)
        inactive = not is_prefill and cfg.decode_target == "hybrid"  # only hybrid decode hands over -1 routes
        gate_up = view.tensors["gate_up"]
        # a float half runs on the fused MoE's grouped GEMM, with its config for this layer's shape
        config = None
        if not (is_quantized(gate_up_type) and is_quantized(down_type)) and (is_prefill or is_resident(layer)):
            from freetoken.moe.fused import try_get_optimal_moe_config

            e, inter_size, hidden = gate_up.shape[0], cfg.local_intermediate, cfg.hidden
            config = try_get_optimal_moe_config((e, 2 * inter_size, hidden), (e, hidden, inter_size), ids.shape[1], x.shape[0])

        def mul_mat_id(w: torch.Tensor, ggml_type: int, a: torch.Tensor) -> torch.Tensor:
            if is_quantized(ggml_type):
                if os.environ.get("FT_DEBUG_MMQ"):
                    _debug_mmq_inputs(w, ggml_type, a, ids, is_prefill)
                return ggml.mul_mat_id(w, ggml_type, a, ids, inactive=inactive)
            # the bf16 kernels read every route's slot: an inactive one reads slot 0 and gets weight 0
            return float_mul_mat_id(w, a, ids.clamp_min(0), topk_weights, config)

        # llama.cpp's build_moe_ffn: each NVFP4 GEMM result times the expert's weight_scale_2 (before
        # its bias), gate and up before the glu, down before the routing weight
        scales = view.tensors.get("global")
        gate_up_scales = scales if scales is not None and gate_up_type == NVFP4 else None
        down_scales = scales if scales is not None and down_type == NVFP4 else None
        # the per-expert biases (gpt-oss) add around the gated activation, as in llama.cpp's build_moe_ffn;
        # the view indexes them by slot id on the decode path and by expert id on prefill / resident
        gate_up_bias = view.tensors.get("gate_up_bias")
        down_bias = view.tensors.get("down_bias")
        glu = fused_glu(layer)
        small = x.shape[0] <= min(ggml.mmvq_max_tokens(t) for t in (gate_up_type, down_type) if is_quantized(t))
        if glu is not None and small and is_quantized(gate_up_type) and gate_up_bias is None:
            gu = (None, None) if gate_up_scales is None else (gate_up_scales[:, 0], gate_up_scales[:, 1])
            if os.environ.get("FT_DEBUG_GLU"):
                _debug_glu_inputs(gate_up, gate_up_type, x, ids)
            inter = ggml.mul_mat_id_glu(gate_up, gate_up_type, x, ids, *glu, *gu)
            if os.environ.get("FT_DEBUG_GLU") and _debug_glu_seen <= 3:
                print("[glu-probe] 5 fused kernel launched, syncing", flush=True)
                torch.cuda.synchronize()
                print("[glu-probe] 5 fused kernel ok", flush=True)
        else:
            h = mul_mat_id(gate_up, gate_up_type, x)
            t, k, two_i = h.shape
            if gate_up_scales is not None:
                h.view(t, k, 2, two_i // 2).mul_(gate_up_scales[ids, :2].unsqueeze(-1))
            if gate_up_bias is not None:
                h.add_(gate_up_bias[ids])
            inter = torch.empty(t, k, two_i // 2, dtype=h.dtype, device=h.device)
            gated_act_and_mul(layer.activation, h.view(-1, two_i), inter.view(-1, two_i // 2), alpha=float(layer.alpha), limit=limit_or_inf(layer))
            del h  # free the gate_up output before the down GEMM allocates its own
        if down_scales is not None and x.shape[0] <= ggml.mmvq_max_tokens(down_type):
            y = ggml.mul_mat_id(view.tensors["down"], down_type, inter, ids, scale=down_scales[:, 2], inactive=inactive)
        else:
            y = mul_mat_id(view.tensors["down"], down_type, inter)
            if down_scales is not None and down_bias is not None:
                # the bias adds after weight_scale_2, so the scale cannot fold into the routing weights
                y.mul_(down_scales[ids, 2].unsqueeze(-1))
            elif down_scales is not None:
                topk_weights = topk_weights * down_scales[ids, 2]
        if down_bias is not None:
            y.add_(down_bias[ids])
        return ggml.moe_weighted_sum(y, topk_weights)

    @staticmethod
    def _apply_float(layer, x, topk_weights, topk_ids, view: ExpertView, ggml_type: int, *, is_prefill: bool):
        """Float experts: their bf16 banks through the bf16 fused MoE (the unquantized kernel's path)."""
        from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

        gate_up, down = view.tensors["gate_up"], view.tensors["down"]
        if gate_up.dtype != x.dtype:
            raise NotImplementedError(f"GGUF {type_name(ggml_type)} experts need {gate_up.dtype} activations, got {x.dtype}")
        impl = fused_experts_impl if is_prefill or is_resident(layer) else fused_experts_decode_impl
        topk_ids = topk_ids.clamp_min(0)  # as in apply: inactive routes read slot 0 with weight 0
        return impl(
            x, gate_up, down, topk_weights, topk_ids, layer.activation, layer.apply_router_weight_on_input,
            float(layer.alpha), limit_or_inf(layer),
            w1_bias=view.tensors.get("gate_up_bias"), w2_bias=view.tensors.get("down_bias"),
        )


@register_method(QuantKind.GGUF, LayerKind.MOE)
class GGUFMoEMethod(MoEMethod):
    candidates = (GgmlMoEKernel,)

    def cpu_spec(self) -> tuple[str | None, dict[str, int]]:
        if self.cfg.has_bias:
            raise NotImplementedError("GGUF experts with per-expert biases (gpt-oss) do not run on the cpu executor")
        gate_up, down = expert_types(self.cfg)
        # a whole-layer float layer (any mix of BF16 / F16 / F32) loads and runs as bf16 banks: the executor's
        # bf16 kernels, which use AVX512-BF16 where the CPU has it (ggml-cpu is built for AVX2 only)
        if float_dtype(gate_up) is not None and float_dtype(down) is not None:
            return "bf16", {}
        # an NVFP4 layer of a file with weight scales reads the layout's global bank too
        scaled = self.cfg.scheme.weight.scale is not None and NVFP4 in (gate_up, down)
        return "gguf_scaled" if scaled else "gguf", {"gate_up_type": gate_up, "down_type": down}


_debug_glu_seen = 0
_debug_mmq_seen = 0


def _debug_mmq_inputs(w, ggml_type, a, ids, is_prefill):
    """FT_DEBUG_MMQ=1: check ggml.mul_mat_id's inputs on the host and raise before a bad launch (diagnostic, 2026-11-03).

    The standalone repro (scripts/ggml_moe_repro.py) passes 140/140 at the real shapes, so a fault in the app means
    these inputs differ: ids range, weight view layout, the zero tail MMQ reads past the last row, or non-finite x.
    """
    global _debug_mmq_seen
    import time

    from freetoken.kernel import ggml as _g

    verbose = _debug_mmq_seen < 6
    t0 = time.perf_counter()
    torch.cuda.synchronize()  # a fault here came from earlier work, not from this GEMM
    if verbose:
        print(f"[mmq-debug] entry sync ok in {time.perf_counter() - t0:.3f}s", flush=True)
    k = a.shape[-1]
    t, top_k = ids.shape
    problems = []
    if not (w.is_cuda and a.is_cuda and ids.is_cuda):
        problems.append(f"device w={w.device} x={a.device} ids={ids.device}")
    if ids.dtype != torch.int32 or not ids.is_contiguous():
        problems.append(f"ids dtype={ids.dtype} contig={ids.is_contiguous()}")
    lo, hi = int(ids.min()), int(ids.max())
    if lo < 0 or hi >= w.shape[0]:
        problems.append(f"ids range [{lo}, {hi}] vs experts {w.shape[0]}")
    if w.dim() != 3 or w.dtype != torch.uint8 or w.shape[2] != row_bytes(ggml_type, k):
        problems.append(f"w shape={tuple(w.shape)} {w.dtype} vs row_bytes {row_bytes(ggml_type, k)} for K={k}")
    if not w.is_contiguous():
        problems.append(f"w not contiguous: stride={w.stride()}")
    if w.data_ptr() % 16:
        problems.append(f"w ptr {w.data_ptr():#x} not 16-aligned")
    if a.shape[0] != t or (a.dim() == 3 and a.shape[1] != top_k):
        problems.append(f"x shape {tuple(a.shape)} vs ids {tuple(ids.shape)}")
    if not bool(torch.isfinite(a).all()):
        problems.append(f"x has {int((~torch.isfinite(a)).sum())} non-finite values")
    tail_info = "-"
    if t > _g.mmvq_max_tokens(ggml_type):  # MMQ reads past the last row; that tail must exist and be zero
        tail = _g._mmq_tail_bytes(ggml_type, k)
        end = w.storage_offset() + w.numel()
        have = w.untyped_storage().nbytes() - end
        if tail and have < tail:
            problems.append(f"MMQ tail: needs {tail} bytes after the last row, storage has {have}")
        elif tail:
            tail_t = torch.empty(0, dtype=torch.uint8, device=w.device).set_(w.untyped_storage(), end, (tail,))
            nz = int(tail_t.count_nonzero())
            tail_info = f"{tail}B nonzero={nz}"
            if nz:
                problems.append(f"MMQ tail has {nz}/{tail} nonzero bytes")
    if verbose or problems:
        _debug_mmq_seen += 1
        print(f"[mmq-debug] prefill={is_prefill} type={ggml_type} w={tuple(w.shape)} stride={w.stride()} ptr={w.data_ptr():#x} "
              f"off={w.storage_offset()} storage={w.untyped_storage().nbytes()} x={tuple(a.shape)} {a.dtype} ids={tuple(ids.shape)} "
              f"range=[{lo},{hi}] tail={tail_info} problems={problems}", flush=True)
    if problems:
        raise RuntimeError("FT_DEBUG_MMQ: " + "; ".join(problems))


def _debug_glu_inputs(gate_up, gate_up_type, x, ids):
    """FT_DEBUG_GLU=1: check the fused-GLU inputs on the host and raise before a bad launch can hang the GPU (diagnostic, 2026-11-03)."""
    global _debug_glu_seen
    if _debug_glu_seen < 3:
        print(f"[glu-debug] entry ids={tuple(ids.shape)} {ids.dtype} {ids.device} ptr={ids.data_ptr():#x} contig={ids.is_contiguous()} "
              f"x={tuple(x.shape)} {x.dtype} {x.device} gate_up={tuple(gate_up.shape)} {gate_up.device}", flush=True)
        import time
        t0 = time.perf_counter()
        torch.cuda.synchronize()  # a fault here came from work before this call (the ids .to(int32) included)
        print(f"[glu-debug] entry sync ok in {time.perf_counter() - t0:.3f}s", flush=True)

        def probe(label, fn):
            print(f"[glu-probe] {time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d} {label} start", flush=True)
            t = time.perf_counter()
            r = fn()
            torch.cuda.synchronize()
            print(f"[glu-probe] {label} ok in {time.perf_counter() - t:.3f}s", flush=True)
            return r

        ids_h = probe("1 ids.cpu (copy, no kernel)", lambda: ids.cpu())
        print(f"[glu-probe] ids values {ids_h.tolist()}", flush=True)
        probe("2 in-place kernel ids.add_(0)", lambda: ids.add_(0))
        probe("3 fresh alloc + fill", lambda: torch.empty(1, dtype=torch.int32, device=ids.device).fill_(0))
        probe("4 ids.min()", lambda: ids.min())
    lo, hi = int(ids.min()), int(ids.max())
    problems = []
    if not (gate_up.is_cuda and x.is_cuda and ids.is_cuda):
        problems.append(f"device gate_up={gate_up.device} x={x.device} ids={ids.device}")
    if lo < 0 or hi >= gate_up.shape[0]:
        problems.append(f"ids range [{lo}, {hi}] vs gate_up slots {gate_up.shape[0]}")
    if not gate_up.is_contiguous():
        problems.append(f"gate_up not contiguous: shape={tuple(gate_up.shape)} stride={gate_up.stride()}")
    if gate_up.data_ptr() % 16:
        problems.append(f"gate_up ptr {gate_up.data_ptr():#x} not 16-aligned")
    if _debug_glu_seen < 3 or problems:
        _debug_glu_seen += 1
        print(f"[glu-debug] type={gate_up_type} gate_up={tuple(gate_up.shape)} {gate_up.dtype} stride={gate_up.stride()} ptr={gate_up.data_ptr():#x} "
              f"x={tuple(x.shape)} {x.dtype} ids={tuple(ids.shape)} range=[{lo},{hi}] problems={problems}", flush=True)
    if problems:
        raise RuntimeError("FT_DEBUG_GLU: " + "; ".join(problems))
