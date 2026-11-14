"""Phase 4 experiment: routed + shared * sigmoid(gate) in one kernel, bit-equal to the three torch bf16 ops.

sigmoid comes from a 64K table that torch fills (its exact bf16 rounding); each product and sum is
rounded to bf16 with explicit round-to-nearest-even, the way torch rounds each op's fp32 result.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


@functools.lru_cache(maxsize=None)
def sigmoid_table(device_index: int) -> torch.Tensor:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("build the sigmoid table before graph capture")
    bits = torch.arange(65536, device=torch.device("cuda", device_index), dtype=torch.int32).to(torch.int16)
    return bits.view(torch.bfloat16).sigmoid()


@triton.jit
def _rne_bf16_bits(x):
    bits = x.to(tl.uint32, bitcast=True)
    r = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return tl.where((bits & 0x7FFFFFFF) > 0x7F800000, (bits >> 16) | 0x40, r)


@triton.jit
def _combine_kernel(routed_ptr, shared_ptr, gate_ptr, table_ptr, out_ptr,
                    stride_r, stride_s, stride_g, stride_o, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    g = tl.load(gate_ptr + row * stride_g).to(tl.int32) & 65535
    sig = tl.load(table_ptr + g).to(tl.float32)
    sh = tl.load(shared_ptr + row * stride_s + offs, mask=mask, other=0.0).to(tl.float32)
    prod_bits = _rne_bf16_bits(sh * sig)
    prod = (prod_bits << 16).to(tl.float32, bitcast=True)
    r = tl.load(routed_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
    out_bits = _rne_bf16_bits(r + prod)
    tl.store(out_ptr.to(tl.pointer_type(tl.uint16)) + row * stride_o + offs, out_bits.to(tl.uint16), mask=mask)


def shared_gate_combine(routed: torch.Tensor, shared: torch.Tensor, gate_logit: torch.Tensor,
                        table: torch.Tensor) -> torch.Tensor:
    """routed + shared * sigmoid(gate_logit); all bf16, routed/shared [M, D], gate_logit [M, 1]."""
    m, d = shared.shape
    out = torch.empty_like(shared)
    _combine_kernel[(m,)](
        routed, shared, gate_logit.view(torch.int16), table, out,
        routed.stride(0), shared.stride(0), gate_logit.stride(0), out.stride(0), d,
        BLOCK=triton.next_power_of_2(d), num_warps=4,
    )
    return out
