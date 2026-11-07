from __future__ import annotations

from typing import TYPE_CHECKING

import os

import torch
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


# FREETOKEN_SHARED_GATE_PAD=0 turns this off. The shared-expert gate is a 1-row GEMV; hipBLAS on gfx1201
# runs that ~2x slower than 16 rows, so the weight is padded with 15 zero rows and column 0 is kept.
# Bit-identical output (ROCm fork, phase 4, 2026-11-07).
_GATE_PAD = os.environ.get("FREETOKEN_SHARED_GATE_PAD", "1") != "0"


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(
        self, config: ModelConfig, hidden_size: int, intermediate_size: int, *, prefix: str = ""
    ):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure as the shared
    expert, so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        super().__init__(config, config.hidden_size, config.intermediate_size, prefix=prefix)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        # routers stay bf16 whatever the checkpoint quantizes
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self._gate_pad_w: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(self._shared_gate(hidden_states))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return (routed + shared).view(num_tokens, hidden_dim)

    def _shared_gate(self, x: torch.Tensor) -> torch.Tensor:
        g = getattr(self.shared_expert_gate, "weight", None)
        if not _GATE_PAD or g is None or g.dim() != 2 or g.shape[0] != 1 or g.dtype != x.dtype:
            return self.shared_expert_gate.forward(x)
        w = self._gate_pad_w
        if w is None or w.device != g.device:
            w = self._gate_pad_w = torch.cat([g, g.new_zeros(15, g.shape[1])], 0).contiguous()
        return torch.nn.functional.linear(x, w)[:, :1]


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
