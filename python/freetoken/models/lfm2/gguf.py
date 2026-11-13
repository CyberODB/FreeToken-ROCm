"""LFM2 / LFM2-MoE from a llama.cpp ``lfm2`` / ``lfm2moe`` GGUF: config from the metadata and the tensor table, tensors by name.

From the on-disk ``lfm2moe`` headers: a layer is a conv layer when it stores ``blk.N.shortconv.*`` (its ``attention.head_count_kv`` entry is 0) and an attention layer when it stores ``blk.N.attn_q``; the final norm is ``token_embd_norm``; the embedding is tied (no ``output.weight``).
Anything this reader does not model (rope scaling, a softmax router, per-layer kv-head counts that differ, extra norms as in ``lfm3``) raises instead of loading.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Iterator

import torch

from freetoken.models.config import ModelConfig
from freetoken.models.gguf.config import GgufConfigShim
from freetoken.models.gguf.reader import gguf_tensor_index
from freetoken.models.gguf.tensors import check_scales, float_tensor, fused, matrix
from freetoken.utils import cached_load_hf_config

from .config import parse_config

_ARCHS = ("lfm2", "lfm2moe")
_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_LEAVES = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "conv.in_proj": "shortconv.in_proj", "conv.out_proj": "shortconv.out_proj",
    "mlp.gate": "ffn_gate_inp",
    **{f"mlp.{p}_proj": f"ffn_{p}" for p in ("gate", "up", "down")},
    **{f"mlp.experts.0.{p}_proj": f"ffn_{p}_exps" for p in ("gate", "up", "down")},
}
# every per-layer tensor this reader consumes; anything else in a block is an unmodeled op
_KNOWN_BLOCK = {
    "attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm",
    "shortconv.conv", "shortconv.in_proj", "shortconv.out_proj",
    "ffn_gate", "ffn_up", "ffn_down", "ffn_gate_inp", "exp_probs_b",
    "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
}
_KNOWN_TOP = {"token_embd", "token_embd_norm", "output"}
_BLOCK = re.compile(r"^blk\.(\d+)\.(.+?)\.(weight|bias|scale|input_scale)$")


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    top = {"model.embed_tokens": "token_embd", "lm_head": "output"}
    if name in top:
        return (top[name],)
    match = _LAYER.match(name)
    if match is None:
        return ()
    leaf = _LEAVES.get(match[2])
    return (f"blk.{match[1]}.{leaf}",) if leaf else ()


def _check_tensors(model_path: str, names) -> None:
    for name in names:
        m = _BLOCK.match(name)
        base = m[2] if m else name.rsplit(".", 1)[0]
        if base not in (_KNOWN_BLOCK if m else _KNOWN_TOP):
            raise NotImplementedError(f"{model_path}: LFM2 GGUF tensor {name!r} is not supported by this reader")


def _hf_config(shim: GgufConfigShim) -> SimpleNamespace:
    md = shim.metadata
    arch = shim.model_type
    if arch not in _ARCHS:
        raise NotImplementedError(f"{shim.model_path}: GGUF architecture {arch!r} is not an LFM2 text model this reader supports")

    def kv(name):
        return md[f"{arch}.{name}"]

    if md.get(f"{arch}.rope.scaling.type", "none") != "none":
        raise NotImplementedError(f"{shim.model_path}: LFM2 GGUF RoPE scaling is not supported")
    check_scales(shim.model_path)
    index = gguf_tensor_index(shim.model_path)
    _check_tensors(shim.model_path, index)
    layers, hidden, heads = int(kv("block_count")), int(kv("embedding_length")), int(kv("attention.head_count"))
    conv_ids = [i for i in range(layers) if f"blk.{i}.shortconv.conv.weight" in index]
    attn_ids = [i for i in range(layers) if f"blk.{i}.attn_q.weight" in index]
    if sorted(conv_ids + attn_ids) != list(range(layers)):
        raise NotImplementedError(f"{shim.model_path}: every LFM2 layer must store either shortconv or attention tensors")
    kv_heads = kv("attention.head_count_kv")
    if isinstance(kv_heads, (list, tuple)):
        if len(kv_heads) != layers or [i for i, n in enumerate(kv_heads) if int(n) == 0] != conv_ids:
            raise NotImplementedError(f"{shim.model_path}: attention.head_count_kv does not match the layers that store shortconv tensors")
        counts = {int(kv_heads[i]) for i in attn_ids}
        if len(counts) != 1:
            raise NotImplementedError(f"{shim.model_path}: attention layers with different kv head counts {sorted(counts)} are not supported")
        num_kv = counts.pop()
    else:
        num_kv = int(kv_heads)
    key_length = md.get(f"{arch}.attention.key_length")
    head_dim = int(key_length) if key_length is not None else hidden // heads
    q_norm = index[f"blk.{attn_ids[0]}.attn_q_norm.weight"] if attn_ids else None
    if q_norm is None or int(q_norm.ne[0]) != head_dim or head_dim * heads != int(index[f"blk.{attn_ids[0]}.attn_q.weight"].ne[1]):
        raise NotImplementedError(f"{shim.model_path}: LFM2 attention geometry does not match head_dim {head_dim} x {heads} heads")
    l_cache = int(kv("shortconv.l_cache"))
    conv = index[f"blk.{conv_ids[0]}.shortconv.conv.weight"] if conv_ids else None
    if conv is None or tuple(int(n) for n in conv.ne) != (l_cache, hidden):
        raise NotImplementedError(f"{shim.model_path}: shortconv.conv must be [{l_cache}, {hidden}] (ggml order)")
    if "token_embd_norm.weight" not in index:
        raise NotImplementedError(f"{shim.model_path}: LFM2 GGUF without token_embd_norm (the final norm) is not supported")
    cfg = dict(
        architectures=list(shim.architectures), model_type=arch,
        num_hidden_layers=layers, hidden_size=hidden,
        intermediate_size=int(kv("feed_forward_length")),
        num_attention_heads=heads, num_key_value_heads=num_kv, head_dim=head_dim,
        vocab_size=shim.vocab_size, max_position_embeddings=int(kv("context_length")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        rope_theta=float(kv("rope.freq_base")),
        conv_layer_ids=conv_ids, conv_l_cache=l_cache,
        tie_word_embeddings=shim.tie_word_embeddings,
    )
    has_experts = any(f"blk.{i}.ffn_gate_exps.weight" in index for i in range(layers))
    if arch == "lfm2moe":
        dense = int(kv("leading_dense_block_count"))
        if int(kv("expert_gating_func")) != 2:
            raise NotImplementedError(f"{shim.model_path}: lfm2moe requires sigmoid routing (expert_gating_func 2)")
        # llama.cpp reads 0.0 as "no scale"
        if float(md.get(f"{arch}.expert_weights_scale", 0.0)) not in (0.0, 1.0):
            raise NotImplementedError(f"{shim.model_path}: lfm2moe expert_weights_scale is not supported")
        for i in range(layers):
            moe = f"blk.{i}.ffn_gate_exps.weight" in index
            if moe != (i >= dense) or moe != (f"blk.{i}.ffn_gate_inp.weight" in index) or moe == (f"blk.{i}.ffn_gate.weight" in index):
                raise NotImplementedError(f"{shim.model_path}: layer {i} does not follow leading_dense_block_count={dense}")
        cfg.update(
            num_experts=int(kv("expert_count")), num_experts_per_tok=int(kv("expert_used_count")),
            moe_intermediate_size=int(kv("expert_feed_forward_length")),
            first_k_dense_replace=dense,
            # the on-disk files carry no expert_weights_norm key; llama.cpp's lfm2moe graph normalizes the selected weights (from memory, unverified)
            norm_topk_prob=bool(md.get(f"{arch}.expert_weights_norm", True)),
        )
    elif has_experts:
        raise NotImplementedError(f"{shim.model_path}: arch lfm2 with routed expert tensors is not supported")
    return SimpleNamespace(**cfg)


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    config = parse_config(_hf_config(shim))
    return replace(config, expert_quant="gguf") if config.moe_enabled else config


def iter_gguf_weights(
    model_path: str, device: torch.device, *, include_moe_experts: bool, include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return
    hf = _hf_config(cached_load_hf_config(model_path))
    index = gguf_tensor_index(model_path)
    conv_ids = set(hf.conv_layer_ids)
    dense = getattr(hf, "first_k_dense_replace", hf.num_hidden_layers)

    def t(name):
        return index[name if name in index else f"{name}.weight"]

    def linear(module, names):
        infos = [t(n) for n in names]
        yield from fused(module, infos, [matrix(v) for v in infos], int(infos[0].ne[0]))

    yield from linear("model.embed_tokens", ["token_embd"])
    if not hf.tie_word_embeddings:
        yield from linear("lm_head", ["output"])
    yield "model.norm.weight", float_tensor(t("token_embd_norm"))
    for layer in range(hf.num_hidden_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        yield f"{pre}.input_layernorm.weight", float_tensor(t(f"{blk}.attn_norm"))
        yield f"{pre}.post_attention_layernorm.weight", float_tensor(t(f"{blk}.ffn_norm"))
        if layer in conv_ids:
            conv = f"{pre}.conv"
            yield from linear(f"{conv}.in_proj", [f"{blk}.shortconv.in_proj"])
            # ggml [l_cache, hidden] -> [hidden, l_cache]
            yield f"{conv}.conv_weight", float_tensor(t(f"{blk}.shortconv.conv"))
            yield from linear(f"{conv}.out_proj", [f"{blk}.shortconv.out_proj"])
        else:
            sa = f"{pre}.self_attn"
            yield from linear(f"{sa}.qkv_proj", [f"{blk}.attn_{p}" for p in ("q", "k", "v")])
            yield from linear(f"{sa}.o_proj", [f"{blk}.attn_output"])
            for p in ("q", "k"):
                yield f"{sa}.{p}_norm.weight", float_tensor(t(f"{blk}.attn_{p}_norm"))
        mlp = f"{pre}.mlp"
        if layer < dense:
            yield from linear(f"{mlp}.gate_up_proj", [f"{blk}.ffn_{p}" for p in ("gate", "up")])
            yield from linear(f"{mlp}.down_proj", [f"{blk}.ffn_down"])
        else:
            yield f"{mlp}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))
            bias = index.get(f"{blk}.exp_probs_b.bias")
            yield f"{mlp}.gate.expert_bias", (
                float_tensor(bias).float() if bias is not None
                else torch.zeros(hf.num_experts, device=device, dtype=torch.float32)
            )


__all__ = ["iter_gguf_weights", "parse_gguf_config", "to_gguf"]
