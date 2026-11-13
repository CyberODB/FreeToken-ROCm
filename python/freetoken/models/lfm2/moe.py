from __future__ import annotations

import torch

from freetoken.kernel.triton.moe_router import fused_topk_sigmoid
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer
from freetoken.models.config import ModelConfig


class Lfm2MoE(BaseOP):
    """lfm2moe routed experts: sigmoid scores, top-k picked on ``score + exp_probs_b``, weights the unbiased scores (FlintMoE without the shared expert)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.gate.expert_bias = torch.zeros(config.num_experts, dtype=torch.float32)
        self.experts = make_moe_layer(
            config, layer_id=layer_id - config.first_k_dense_replace,
            quant_config=config.quant, prefix=f"{prefix}.experts",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, ids = fused_topk_sigmoid(
            self.gate.forward(x).float(), self.gate.expert_bias, self.top_k, self.renormalize,
        )
        return self.experts.routed_forward(x, weights, ids)


__all__ = ["Lfm2MoE"]
