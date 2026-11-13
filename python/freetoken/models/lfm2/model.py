"""LFM2 / LFM2-MoE: pre-norm blocks whose operator is a short conv or GQA (qk-norm, RoPE), SwiGLU FFN or routed experts, and the embedding norm as the final norm."""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel, GatedMLP
from freetoken.models.config import ModelConfig
from freetoken.models.qwen3.attention import Qwen3Attention
from freetoken.utils import nvtx_annotate

from .moe import Lfm2MoE
from .shortconv import Lfm2ShortConv


class Lfm2DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self.is_conv = config.is_linear_layer(layer_id)
        if self.is_conv:
            group = config.linear_attention_group()
            self.conv = Lfm2ShortConv(
                config.hidden_size, group.conv_kernel_dim, layer_id,
                quant_config=config.quant, prefix=f"{prefix}.conv",
            )
        else:
            self.self_attn = Qwen3Attention(config, layer_id, has_qk_norm=True, prefix=f"{prefix}.self_attn")
        self.mlp = (
            Lfm2MoE(config, layer_id, prefix=f"{prefix}.mlp")
            if config.moe_enabled and layer_id >= config.first_k_dense_replace
            else GatedMLP(config, quant_config=config.quant, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.conv.forward(x) if self.is_conv else self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        return self.mlp.forward(x), residual


class Lfm2Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, quant_config=config.quant, prefix=f"{prefix}.embed_tokens",
        )
        self.layers = OPList([
            Lfm2DecoderLayer(config, i, prefix=f"{prefix}.layers.{i}") for i in range(config.num_layers)
        ])
        # GGUF token_embd_norm: applied to the last hidden state
        self.norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x, residual = self.embed_tokens.forward(input_ids), None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Lfm2ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Lfm2Model(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant, prefix="lm_head",
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Lfm2ForCausalLM"]
