# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 MiniMax AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import copy
import math
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import CrossEntropyLoss

from ..activations import ACT2FN
from ..conversion_utils import StateDictNameMapping, init_name_mappings
from ..model_outputs import (
    MoECausalLMOutputWithPast,
    MoEModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ..model_utils import PretrainedModel, register_base_model
from .configuration import MiniMaxText01Config

__all__ = [
    "MiniMaxText01PreTrainedModel",
    "MiniMaxText01Model",
    "MiniMaxText01ForCausalLM",
    "MiniMaxText01ForSequenceClassification",
]

BLOCK = 256


def get_activation_fn(activation):
    print(f"activation: {activation}")
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":

        def f(x):
            with paddle.no_grad():
                x_max = paddle.max(x, axis=-1, keepdim=True)
            y = paddle.exp(x - x_max)
            return y

        return f
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":

        def f(x):
            return 1 + F.elu(x)

        return f
    elif activation == "2+elu":

        def f(x):
            return 2 + F.elu(x)

        return f
    elif activation == "silu" or activation == "swish":
        return F.silu
    elif activation == "sine":
        return paddle.sin
    else:
        return lambda x: x


class MiniMaxText01RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super(MiniMaxText01RMSNorm, self).__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size], dtype="float32", default_initializer=paddle.nn.initializer.Constant(1.0)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = paddle.cast(hidden_states, dtype="float32")

        variance = paddle.mean(paddle.square(hidden_states), axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)

        return self.weight * paddle.cast(hidden_states, dtype=input_dtype)


def load_balancing_loss_func(
    gate_logits: paddle.Tensor, num_experts: paddle.Tensor = None, top_k=2, attention_mask: paddle.Tensor = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in PaddlePaddle.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[paddle.Tensor, Tuple[paddle.Tensor]]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (paddle.Tensor, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0.0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].place
        concatenated_gate_logits = paddle.concat([layer_gate.to(compute_device) for layer_gate in gate_logits], axis=0)

    routing_weights = F.softmax(concatenated_gate_logits, axis=-1)

    _, selected_experts = paddle.topk(routing_weights, top_k, axis=-1)

    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)

    if attention_mask is None:
        tokens_per_expert = paddle.mean(expert_mask.astype(paddle.float32), axis=0)

        router_prob_per_expert = paddle.mean(routing_weights, axis=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape([-1, top_k, num_experts])
            .to(compute_device)
        )

        tokens_per_expert = paddle.sum(
            expert_mask.astype(paddle.float32) * expert_attention_mask, axis=0
        ) / paddle.sum(expert_attention_mask, axis=0)

        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape([-1, num_experts])
            .to(compute_device)
        )

        router_prob_per_expert = paddle.sum(routing_weights * router_per_expert_attention_mask, axis=0) / paddle.sum(
            router_per_expert_attention_mask, axis=0
        )

    overall_loss = paddle.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# Copied from a similar function in transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0, dtype="int32"), [1, 0])
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class MiniMaxText01LightningAttention(nn.Layer):
    def __init__(self, config: MiniMaxText01Config, layer_idx: Optional[int] = None):
        super().__init__()
        paddle.seed(42)  # add
        bias = False
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

        self.out_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias_attr=bias)
        self.act = get_activation_fn(config.hidden_act)
        self.norm = MiniMaxText01RMSNorm(self.head_dim * self.num_heads)

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.head_dim * self.num_heads, bias_attr=bias)

        self.output_gate = nn.Linear(self.hidden_size, self.head_dim * self.num_heads, bias_attr=bias)

        # for inference only
        self.offset = 0
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor] = None,  # (b, h, n, m)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[Tensor]] = None,
        use_cache: bool = False,
        slope_rate: Optional[Tensor] = None,
        **kwargs
    ):
        if not self.training:
            return self.inference(
                hidden_states,
                attn_mask,
                output_attentions,
                past_key_value,
                use_cache,
                slope_rate,
            )

    def inference(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,  # (b, n)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[Tensor]] = None,
        use_cache: bool = False,
        slope_rate: Optional[Tensor] = None,  # (h, 1, 1)
    ):
        b, n, d = x.shape
        qkv = self.act(self.qkv_proj(x))
        new_shape = list(qkv.shape[:-1]) + [self.num_heads, -1]
        qkv = qkv.reshape(new_shape)
        q, k, v = paddle.split(qkv, [self.head_dim] * 3, axis=-1)
        q = paddle.transpose(q, perm=[0, 2, 1, 3])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        v = paddle.transpose(v, perm=[0, 2, 1, 3])

        if past_key_value is None:
            self.offset = q.shape[-2]
        else:
            self.offset += 1

        ratio = paddle.exp(-slope_rate)
        if past_key_value is None:
            slope_rate = paddle.cast(slope_rate, dtype="float32")
            if attn_mask is not None:
                v = paddle.masked_fill(v, (1 - attn_mask).unsqueeze(1).unsqueeze(-1).astype(paddle.bool), 0)

            NUM_BLOCK = (n + BLOCK - 1) // BLOCK
            b, h, n, d = q.shape
            e = v.shape[-1]
            array = paddle.arange(BLOCK).to(q.place).astype("float32") + 1
            q_decay = paddle.exp(-slope_rate * array.reshape([-1, 1]))
            k_decay = paddle.exp(-slope_rate * (BLOCK - array.reshape([-1, 1])))
            index = array[:, None] - array[None, :]

            s_index = slope_rate * paddle.unsqueeze(paddle.unsqueeze(index, axis=0), axis=0)
            s_index = paddle.where(index >= 0, -s_index, float("-inf"))
            diag_decay = paddle.exp(s_index)

            kv = paddle.zeros([b, h, d, e], dtype="float32")
            kv = kv.to(q.place)
            output = paddle.empty([b, h, n, e], dtype=q.dtype)
            output = output.to(q.place)
            for i in range(NUM_BLOCK):
                si = i * BLOCK
                ei = min(si + BLOCK, n)
                m = ei - si
                qi = q[:, :, si:ei]
                ki = k[:, :, si:ei]
                vi = v[:, :, si:ei]

                qkv_none_diag = paddle.matmul(qi * q_decay[:, :m], kv).to("float32")

                qk = (
                    paddle.matmul(qi, paddle.transpose(ki, perm=[0, 1, 3, 2])).astype(paddle.float32)
                    * diag_decay[:, :, :m, :m]
                )

                qkv_diag = paddle.matmul(qk.astype(paddle.float32), vi.astype(paddle.float32))

                block_decay = paddle.exp(-slope_rate * m)
                output[:, :, si:ei] = qkv_none_diag + qkv_diag
                kv = block_decay * kv + paddle.matmul((ki * k_decay[:, -m:]).transpose([0, 1, 3, 2]), vi)

        else:
            kv = past_key_value
            output = []
            for i in range(n):
                kv = ratio * kv + paddle.einsum(
                    "... n d, ... n e -> ... d e",
                    k[:, :, i : i + 1],
                    v[:, :, i : i + 1],
                )
                qkv = paddle.einsum("... n e, ... e d -> ... n d", q[:, :, i : i + 1], kv)
                output.append(qkv)
            output = paddle.concat(output, axis=-2)

        output = paddle.transpose(output, perm=[0, 2, 1, 3])
        output = output.reshape([b, n, -1])

        output = self.norm(output)

        output = F.sigmoid(self.output_gate(x)) * output
        output = self.out_proj(output)

        attn_weights = None

        return output, attn_weights, kv


class MiniMaxText01RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (paddle.arange(0, self.dim, 2, dtype="float32") / self.dim))
        self.inv_freq = paddle.to_tensor(inv_freq, stop_gradient=True)

        # Build here to make `paddle.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.place, dtype="float32")

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached, dtype="int64")
        t = t.astype("float32")

        freqs = paddle.outer(t, self.inv_freq)
        emb = paddle.concat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().astype(dtype)
        self.sin_cached = emb.sin().astype(dtype)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.place, dtype="float32")

        return (
            self.cos_cached[:seq_len].astype("float32"),
            self.sin_cached[:seq_len].astype("float32"),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    dtype = q.dtype
    rot_dim = cos.shape[-1]
    q_, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    # Unsqueeze the cosine and sine tensors
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    # Apply rotary embedding using the custom rotate_half function
    q_embed = (q_ * cos) + (rotate_half(q_) * sin)
    k_embed = (k_ * cos) + (rotate_half(k_) * sin)

    # Concatenate q_embed and k_embed with their respective pass tensors and convert to the correct dtype
    return paddle.concat((q_embed, q_pass), axis=-1).astype(dtype), paddle.concat((k_embed, k_pass), axis=-1).astype(
        dtype
    )


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(x, axis=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


class Cache:
    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, new_keys: paddle.Tensor, new_values: paddle.Tensor):
        if self.keys is None:
            self.keys = new_keys
            self.values = new_values
        else:
            self.keys = paddle.concat([self.keys, new_keys], axis=-2)
            self.values = paddle.concat([self.values, new_values], axis=-2)

    def get(self):
        return self.keys, self.values

    def get_usable_length(self, kv_seq_len: int, layer_idx: int) -> int:
        if self.keys is None:
            return 0
        return self.keys.shape[-2]

    def clear(self):
        self.keys = None
        self.values = None


class MiniMaxText01Attention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MiniMaxText01Config, layer_idx: Optional[int] = None):
        super(MiniMaxText01Attention, self).__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=False)

        self.rotary_dim = getattr(config, "rotary_dim", self.head_dim)

        self.rotary_emb = MiniMaxText01RotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:

        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value[0].shape[-3]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=-2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=-2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)

        if attention_mask is not None:

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, axis=-1).astype(paddle.float32)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = paddle.matmul(attn_weights, value_states)

        if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"attn_output should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([bsz, q_len, -1])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MiniMaxText01MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MiniMaxText01BlockSparseTop2MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias_attr=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias_attr=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias_attr=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MiniMaxText01BLockSparseTop2MLP(MiniMaxText01BlockSparseTop2MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MiniMaxText01SparseMoeBlock(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias_attr=False)

        self.experts = nn.LayerList([MiniMaxText01BlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if self.training and self.jitter_noise > 0:
            hidden_states *= (
                paddle.rand_like(hidden_states, dtype=hidden_states.dtype) * (1.0 + self.jitter_noise)
                - self.jitter_noise
            )
        hidden_states = hidden_states.reshape((-1, hidden_dim))

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, axis=1)
        routing_weights, selected_experts = paddle.topk(routing_weights, self.top_k, axis=-1)
        routing_weights /= paddle.sum(routing_weights, axis=-1, keepdim=True)
        routing_weights = routing_weights.astype(hidden_states.dtype)

        final_hidden_states = paddle.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype).to(
            hidden_states.place
        )

        expert_mask = paddle.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).transpose([2, 1, 0])

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            indices = paddle.nonzero(expert_mask[expert_idx])
            idx = indices[:, 0]
            top_x = indices[:, 1]

            if top_x.shape[0] == 0:
                continue
            else:
                current_state = hidden_states[top_x].reshape((-1, hidden_dim))
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states = paddle.index_add(
                x=final_hidden_states,
                index=top_x,
                value=current_hidden_states.astype(final_hidden_states.dtype),
                axis=0,
            )

        final_hidden_states = final_hidden_states.reshape((batch_size, sequence_length, hidden_dim))
        return final_hidden_states, router_logits


class MiniMaxText01DecoderLayer(nn.Layer):
    def __init__(self, config, layer_idx: int):
        super(MiniMaxText01DecoderLayer, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = self.build_attn(config, layer_idx)
        self.layer_idx = layer_idx

        self.block_sparse_moe = MiniMaxText01SparseMoeBlock(config)
        self.input_layernorm = MiniMaxText01RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxText01RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.postnorm = getattr(config, "postnorm", False)
        if config.attention_type == 0:
            self.layernorm_attention_alpha = getattr(config, "layernorm_linear_attention_alpha", 1)
            self.layernorm_attention_beta = getattr(config, "layernorm_linear_attention_beta", 1)
        else:
            self.layernorm_attention_alpha = getattr(config, "layernorm_full_attention_alpha", 1)
            self.layernorm_attention_beta = getattr(config, "layernorm_full_attention_beta", 1)

        self.layernorm_mlp_alpha = getattr(config, "layernorm_mlp_alpha", 1)
        self.layernorm_mlp_beta = getattr(config, "layernorm_mlp_beta", 1)

        shared_intermediate = getattr(config, "shared_intermediate_size", 0)
        self.shared_moe = False
        if shared_intermediate > 0:
            self.shared_moe = True
            self.shared_mlp = MiniMaxText01MLP(config)
            self.coefficient = nn.Linear(self.hidden_size, 1, bias_attr=False)

    def build_attn(self, config, layer_idx):
        if config.attention_type == 0:
            Attention_module = MiniMaxText01LightningAttention
        else:
            # Attention_module = MiniMaxText01FlashAttention2
            Attention_module = MiniMaxText01Attention
        return Attention_module(config, layer_idx)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        slope_rate: Optional[float] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attn_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            slope_rate=slope_rate,
        )

        hidden_states = residual * self.layernorm_attention_alpha + hidden_states * self.layernorm_attention_beta

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states

        moe_hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        if self.shared_moe:
            output_mlp = self.shared_mlp(hidden_states)
            weight_fp32 = self.coefficient.weight.astype("float32")
            hidden_states_fp32 = hidden_states.astype("float32")
            coef = paddle.matmul(hidden_states_fp32, weight_fp32, transpose_y=True)
            coef = F.sigmoid(coef).astype(hidden_states.dtype)
            hidden_states = moe_hidden_states * (1 - coef) + output_mlp * coef
        else:
            hidden_states = moe_hidden_states

        hidden_states = residual * self.layernorm_mlp_alpha + hidden_states * self.layernorm_mlp_beta

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class MiniMaxText01PreTrainedModel(PretrainedModel):
    config_class = MiniMaxText01Config
    base_model_prefix = "model"

    @classmethod
    def _get_name_mappings(cls, config: MiniMaxText01Config) -> list[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]

        for layer_index in range(config.num_hidden_layers):
            if config.attn_type_list[layer_index] == 0:  # Lightning Attention
                layer_mappings = [
                    [f"layers.{layer_index}.self_attn.out_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.norm.weight"],
                    [f"layers.{layer_index}.self_attn.qkv_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.output_gate.weight", None, "transpose"],
                ]
            else:  # Softmax Attention
                layer_mappings = [
                    [f"layers.{layer_index}.self_attn.q_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.k_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.v_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                ]

            # MoE gate and experts
            layer_mappings.append([f"layers.{layer_index}.block_sparse_moe.gate.weight", None, "transpose"])

            for expert_idx in range(config.num_local_experts):
                layer_mappings.extend(
                    [
                        [f"layers.{layer_index}.block_sparse_moe.experts.{expert_idx}.w1.weight", None, "transpose"],
                        [f"layers.{layer_index}.block_sparse_moe.experts.{expert_idx}.w2.weight", None, "transpose"],
                        [f"layers.{layer_index}.block_sparse_moe.experts.{expert_idx}.w3.weight", None, "transpose"],
                    ]
                )

            layer_mappings.extend(
                [
                    [f"layers.{layer_index}.input_layernorm.weight"],
                    [f"layers.{layer_index}.post_attention_layernorm.weight"],
                ]
            )

            model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)

        for mapping in model_mappings:
            mapping[0] = "model." + mapping[0]
            mapping[1] = "model." + mapping[1]
        model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        # if "MiniMaxText01Model" not in config.architectures:
        #     for mapping in model_mappings:
        #         mapping[0] = "model." + mapping[0]
        #         mapping[1] = "minimax_text01." + mapping[1]

        #     if not config.tie_word_embeddings:
        #         model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    def _init_weights(self, layer):
        pass


@register_base_model
class MiniMaxText01Model(MiniMaxText01PreTrainedModel):
    def __init__(self, config: MiniMaxText01Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.attn_type_list = config.attn_type_list
        config_copy = copy.deepcopy(config)

        self.layers = nn.LayerList([])

        for i in range(config.num_hidden_layers):
            _config = copy.deepcopy(config)
            if self.attn_type_list[i] == 0:
                _config._attn_implementation = "linear_attention"
                _config.attention_type = 0
            else:
                _config._attn_implementation = config_copy._attn_implementation
                _config.attention_type = 1
            self.layers.append(MiniMaxText01DecoderLayer(_config, i))

        self._attn_implementation = config_copy._attn_implementation
        self.norm = MiniMaxText01RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.slopes = self._build_slope_tensor(config.num_attention_heads)
        # mask
        self._linear_attn_mask = paddle.empty([0])

        # Initialize weights and apply final processing
        # self.post_init()

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = paddle.to_tensor(get_slopes(n_attention_heads), dtype=paddle.float32).reshape(
            [n_attention_heads, 1, 1]
        )

        return slopes

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoEModelOutputWithPast]:  # Update return type to MoEModelOutputWithPast
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            default_device = input_ids.place
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            default_device = inputs_embeds.place
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        seq_length_with_past = seq_length
        if past_key_values is not None:
            for idx in range(len(past_key_values)):
                if self.attn_type_list[idx] == 1:
                    past_key_values_length = past_key_values[idx][0].shape[-3]
                    seq_length_with_past = seq_length_with_past + past_key_values_length
                    break

        if position_ids is None:
            position_ids = paddle.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=paddle.int64
            ).to(input_ids.place)
            position_ids = position_ids.unsqueeze(0).reshape([-1, seq_length])
        else:
            position_ids = position_ids.reshape([-1, seq_length]).astype("int64")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        slope_rates = [self.slopes.to(default_device) for _ in range(len(self.layers))]
        hidden_states = inputs_embeds
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            attn_mask = attention_mask
            slope_rate = slope_rates[idx]
            slope_rate = slope_rate * (1 - idx / (len(self.layers) - 1) + 1e-5)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                slope_rate=slope_rate,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class MiniMaxText01ForCausalLM(MiniMaxText01PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniMaxText01Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        """
        Forward pass for MiniMaxText01ForCausalLM in Paddle
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder outputs consist of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)  # Ensure logits are in the correct dtype.

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.reshape((-1, self.config.vocab_size))
            shift_labels = shift_labels.reshape((-1))
            # Enable model parallelism
            shift_labels = shift_labels.astype(shift_logits.dtype)
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.astype(
                    loss.dtype
                )  # Ensure loss is on the same device and dtype

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        paddle.device.cuda.empty_cache()  # Clearing the cache on GPU
        return MoECausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if inputs_embeds are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.astype(past_state.dtype)) for past_state in layer_past),
            )
        return reordered_past


class MiniMaxText01ForSequenceClassification(MiniMaxText01PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MiniMaxText01Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias_attr=False)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        Args:
            input_ids (`paddle.LongTensor` of shape `(batch_size,)`, *optional*):
                Input tensor containing token ids for the input sequence.
            labels (`paddle.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Transformer outputs consist of (hidden_states, ...)
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = paddle.equal(input_ids, self.config.pad_token_id).astype("int32").argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths
            else:
                sequence_lengths = -1

        # pooled_logits = logits[paddle.arange(batch_size), sequence_lengths]
        pooled_logits = logits.gather_nd(paddle.stack([paddle.arange(logits.shape[0]), sequence_lengths], axis=-1))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
