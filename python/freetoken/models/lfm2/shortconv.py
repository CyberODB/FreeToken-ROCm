"""LFM2 short-range gated conv block (llama.cpp ``blk.N.shortconv.*``).

    B, C, x = in_proj(h).chunk(3)
    y       = out_proj(C * conv(B * x))

``conv`` is a causal depthwise 1D conv over the last ``l_cache`` positions with no bias and no activation, so the fused GatedDeltaNet conv kernels (which apply silu) do not fit and the conv is plain torch ops.
Its left context (the last ``l_cache - 1`` conv inputs of each request) is the per-request slot state ``lfm2_conv`` on the LinearStatePool, ``[num_slots, hidden, l_cache - 1]`` with time ascending -- the same layout as the pool's own GDN conv state, so the hybrid-radix track snapshot is the GDN one.

Not verified against a reference run: the chunk order (B, C, x) and the kernel orientation (tap ``l_cache - 1`` multiplies the current token) are from memory of HF ``Lfm2ShortConvLayer`` / llama.cpp ``ggml_ssm_conv``.
"""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.layers.quantization import QuantConfig
from freetoken.utils import nvtx_annotate

LFM2_CONV_STATE = "lfm2_conv"


def conv_decode(bx: torch.Tensor, weight: torch.Tensor, states: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """One token per request. ``bx`` [B, C], ``weight`` [C, K], ``states`` [num_slots, C, K-1], ``slots`` [B] int64; rolls the state in place."""
    window = torch.cat([states.index_select(0, slots).to(bx.dtype), bx.unsqueeze(-1)], dim=-1)  # [B, C, K]
    states.index_copy_(0, slots, window[..., 1:].to(states.dtype))
    return (window * weight).sum(-1)


def conv_prefill(
    bx: torch.Tensor,
    weight: torch.Tensor,
    states: torch.Tensor,
    slots: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor | None,
) -> torch.Tensor:
    """Varlen causal conv over the flat token rows of several requests.

    ``bx`` [T, C]; ``cu_seqlens`` [R+1]; ``slots`` [R] int64. A request without initial state starts from zeros; every request's slot ends holding its last K-1 inputs."""
    total, km1 = bx.shape[0], states.shape[-1]
    cu = cu_seqlens.to(torch.int64)
    nreq = cu.numel() - 1
    init = states.index_select(0, slots).to(bx.dtype)  # [R, C, K-1]
    if has_initial_state is not None:
        init = init * has_initial_state.to(bx.dtype).view(-1, 1, 1)
    # ext: per request, its K-1 state rows followed by its token rows
    rows = torch.arange(total, device=bx.device)
    req = torch.bucketize(rows, cu[1:], right=True)
    tok_pos = rows + (req + 1) * km1
    ridx = torch.arange(nreq, device=bx.device)
    state_pos = (cu[:-1] + ridx * km1).unsqueeze(1) + torch.arange(km1, device=bx.device)  # [R, K-1]
    ext = bx.new_empty(total + nreq * km1, bx.shape[1])
    ext.index_copy_(0, state_pos.reshape(-1), init.transpose(1, 2).reshape(nreq * km1, -1))
    ext.index_copy_(0, tok_pos, bx)
    out = ext.index_select(0, tok_pos) * weight[:, km1]
    for k in range(km1):
        out = out + ext.index_select(0, tok_pos - (km1 - k)) * weight[:, k]
    # the last K-1 ext rows of each request (state rows when it has fewer tokens)
    end = cu[1:] + (ridx + 1) * km1
    tail = (end.unsqueeze(1) - km1 + torch.arange(km1, device=bx.device)).reshape(-1)
    new_state = ext.index_select(0, tail).view(nreq, km1, -1).transpose(1, 2)
    states.index_copy_(0, slots, new_state.to(states.dtype))
    return out


class Lfm2ShortConv(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        l_cache: int,
        layer_id: int,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        if l_cache < 2:
            raise NotImplementedError(f"LFM2 shortconv.l_cache={l_cache}: expected a kernel of at least 2")
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.l_cache = l_cache
        self.in_proj = LinearReplicated(
            hidden_size, 3 * hidden_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.in_proj",
        )
        # depthwise kernel [hidden, l_cache]; tap l_cache-1 multiplies the current token
        self.conv_weight = torch.empty(hidden_size, l_cache)
        self.out_proj = LinearReplicated(
            hidden_size, hidden_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.out_proj",
        )

    def _states(self, pool) -> torch.Tensor:
        assert pool is not None and pool.has_slot_state(LFM2_CONV_STATE), (
            "LFM2 conv layers need the lfm2_conv slot state (ModelConfig.slot_states)"
        )
        return pool.slot_state(LFM2_CONV_STATE, self.layer_id)

    @nvtx_annotate("ShortConv")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device, pool)
            batch.fla_metadata = fla

        b, c, x = self.in_proj.forward(hidden_states).chunk(3, dim=-1)
        bx = b * x
        states = self._states(pool)
        weight = self.conv_weight.to(bx.dtype)
        slots = fla.cache_indices.to(torch.int64)
        if not batch.is_prefill:
            if batch.phase == "verify" or bx.shape[0] != slots.numel():
                # the engine rejects speculative decoding at config time (lfm2_conv declares no verify handler)
                raise NotImplementedError("LFM2 conv layers serve one decode row per request; speculative verify is not implemented")
            conv_out = conv_decode(bx, weight, states, slots)
        else:
            conv_out = conv_prefill(bx, weight, states, slots, fla.cu_seqlens, fla.has_initial_state)
            if fla.track_dst is not None:
                # hybrid-radix checkpoint: the K-1 conv inputs ending at the chunk boundary, as GatedDeltaNet._write_track_snapshot
                win = bx.index_select(0, fla.track_conv_src.reshape(-1)).view(*fla.track_conv_src.shape, -1).transpose(-1, -2)
                states.index_copy_(0, fla.track_dst, win.contiguous().to(states.dtype))
        return self.out_proj.forward(c * conv_out)


__all__ = ["LFM2_CONV_STATE", "Lfm2ShortConv", "conv_decode", "conv_prefill"]
