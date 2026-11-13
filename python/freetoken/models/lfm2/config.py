"""ModelConfig of the LFM2 family (llama.cpp ``lfm2`` / ``lfm2moe``): conv layers and GQA layers interleaved.

The conv layers ride the LinearStatePool: a placeholder linear group marks them as layers without paged KV (its own GDN state is 3 values per slot and layer and unused), and the conv left context is the declared slot state ``lfm2_conv``.
"""

from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SlotStateSpec,
)

from .shortconv import LFM2_CONV_STATE


def parse_config(hf: Any) -> ModelConfig:
    """``hf`` carries the fields ``gguf._hf_config`` builds from the GGUF metadata."""
    layers = int(hf.num_hidden_layers)
    conv_ids = tuple(int(i) for i in hf.conv_layer_ids)
    attn_ids = tuple(i for i in range(layers) if i not in set(conv_ids))
    if not conv_ids or not attn_ids:
        raise NotImplementedError("LFM2 requires both conv and attention layers")
    hidden, head_dim, l_cache = int(hf.hidden_size), int(hf.head_dim), int(hf.conv_l_cache)
    if l_cache < 2:
        raise NotImplementedError(f"LFM2 shortconv.l_cache={l_cache}: expected a kernel of at least 2")
    rope = RotaryConfig(
        head_dim=head_dim, rotary_dim=head_dim,
        max_position=int(hf.max_position_embeddings), base=float(hf.rope_theta), scaling=None,
    )
    full = FullAttentionGroupConfig(
        name="full", layer_ids=attn_ids, num_kv_heads=int(hf.num_key_value_heads),
        head_dim=head_dim, rotary_config=rope,
    )
    # placeholder geometry: only layer_ids and conv_kernel_dim (the track window length) are used
    conv = LinearGatedDeltaGroupConfig(
        name="conv", layer_ids=conv_ids, num_key_heads=1, num_value_heads=1,
        key_head_dim=1, value_head_dim=1, conv_kernel_dim=l_cache, output_gate="silu",
    )
    num_experts = int(getattr(hf, "num_experts", 0) or 0)
    return ModelConfig(
        num_layers=layers, num_qo_heads=int(hf.num_attention_heads),
        num_kv_heads=int(hf.num_key_value_heads), head_dim=head_dim,
        hidden_size=hidden, vocab_size=int(hf.vocab_size),
        intermediate_size=int(hf.intermediate_size), hidden_act="silu",
        rms_norm_eps=float(hf.rms_norm_eps),
        tie_word_embeddings=bool(hf.tie_word_embeddings), rotary_config=rope,
        num_experts=num_experts,
        num_experts_per_tok=int(getattr(hf, "num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(getattr(hf, "moe_intermediate_size", 0) or 0),
        norm_topk_prob=bool(getattr(hf, "norm_topk_prob", False)),
        model_type=hf.model_type, architectures=list(hf.architectures),
        moe_enabled=num_experts > 0,
        first_k_dense_replace=int(getattr(hf, "first_k_dense_replace", 0) or 0),
        use_qk_norm=True,
        attention_groups=tuple(sorted((full, conv), key=lambda g: g.layer_ids[0])),
        # verify=None: the engine refuses speculative decoding for this model at config time
        slot_states=(SlotStateSpec(LFM2_CONV_STATE, (hidden, l_cache - 1), layer_ids=conv_ids),),
    )


__all__ = ["parse_config"]
