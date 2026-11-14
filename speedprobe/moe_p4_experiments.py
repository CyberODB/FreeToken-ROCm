from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import os

from freetoken.kernel.graph_stamps import fine as _fine

_MERGED_GATE = os.environ.get("FT_P4_MERGED_GATE", "")  # "1": router+gate one GEMV; "2": gate padded alone
_GATE_CHECK = os.environ.get("FT_P4_GATE_CHECK") == "1"
_FUSED_COMBINE = os.environ.get("FT_P4_FUSED_COMBINE") == "1"  # routed + shared*sigmoid(gate) in one kernel
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
        self._fine_lid = -1 if layer_id is None else layer_id
        self._router_gate_w: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        lid = self._fine_lid
        if _MERGED_GATE == "2":
            # bit-identical variant: router unchanged; the 1-row gate GEMV padded to 16 rows (zeros) picks a faster kernel
            w = self._router_gate_w
            if w is None or w.dtype != hidden_states.dtype:
                g = self.shared_expert_gate.weight
                w = self._router_gate_w = torch.cat([g, g.new_zeros(15, g.shape[1])], 0).to(hidden_states.dtype).contiguous()
            router_logits = self.gate.forward(hidden_states)
            gate_logit = torch.nn.functional.linear(hidden_states, w)[:, :1]
            if _GATE_CHECK and not torch.cuda.is_current_stream_capturing() and self._fine_lid < 3:
                g0 = self.shared_expert_gate.forward(hidden_states)
                print(f"[gate-check2] L{self._fine_lid} M={hidden_states.shape[0]} gate_eq {torch.equal(g0, gate_logit)}", flush=True)
        elif _MERGED_GATE == "1":
            # router + shared-expert gate as one GEMV: hipBLAS runs a 1-row GEMV ~2x slower than 129 rows (phase 4)
            w = self._router_gate_w
            if w is None or w.dtype != hidden_states.dtype:
                w = self._router_gate_w = torch.cat(
                    [self.gate.weight, self.shared_expert_gate.weight], 0).to(hidden_states.dtype).contiguous()
            logits = torch.nn.functional.linear(hidden_states, w)
            n = self.gate.weight.shape[0]
            router_logits, gate_logit = logits[:, :n], logits[:, n:]
            if _GATE_CHECK and not torch.cuda.is_current_stream_capturing() and self._fine_lid < 3:
                r0, g0 = self.gate.forward(hidden_states), self.shared_expert_gate.forward(hidden_states)
                print(f"[gate-check] L{self._fine_lid} M={hidden_states.shape[0]} w {self.gate.weight.dtype} "
                      f"x {hidden_states.dtype} router_eq {torch.equal(r0, router_logits)} gate_eq {torch.equal(g0, gate_logit)} "
                      f"router_maxdiff {(r0.float() - router_logits.float()).abs().max().item():.3g}", flush=True)
        else:
            router_logits = self.gate.forward(hidden_states)
            gate_logit = None
        _fine("router", lid)
        shared = self.shared_expert.forward(hidden_states)
        _fine("shared expert", lid)
        if gate_logit is None:
            gate_logit = self.shared_expert_gate.forward(hidden_states)
        table = None
        if _FUSED_COMBINE and shared.dtype == torch.bfloat16 and gate_logit.dtype == torch.bfloat16:
            from freetoken.kernel.triton.shared_gate import sigmoid_table, shared_gate_combine

            if not torch.cuda.is_current_stream_capturing() or sigmoid_table.cache_info().currsize:
                table = sigmoid_table(shared.device.index if shared.device.index is not None else torch.cuda.current_device())
        if table is None:
            shared = shared * torch.sigmoid(gate_logit)
        _fine("shared gate (gemv+sigmoid+mul)", lid)
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        _fine("routed tail (after offload stamps)", lid)
        if table is not None and routed.dtype == torch.bfloat16 and routed.stride(-1) == 1 and shared.stride(-1) == 1:
            out = shared_gate_combine(routed, shared, gate_logit, table)
            if _GATE_CHECK and not torch.cuda.is_current_stream_capturing() and lid < 3:
                ref = routed + shared * torch.sigmoid(gate_logit)
                print(f"[combine-check] L{lid} M={num_tokens} eq {torch.equal(ref, out)}", flush=True)
            return out.view(num_tokens, hidden_dim)
        if table is not None:
            shared = shared * torch.sigmoid(gate_logit)
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
