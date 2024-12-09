# !/usr/bin/env python3

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Ernie35 model"""
import contextlib
import math
import os
from functools import partial
from typing import Optional, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from configuration import Ernie35Config
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu.mp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.nn.layer.fused_dropout_add import FusedDropoutAdd

from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model
from paddlenlp.utils.log import logger

try:
    from paddle.nn.functional.flash_attention import flash_attention

    logger.warning("Use flash attention in scaled-dot-product. Attention mask is deprecated")
except:
    flash_attention = None

try:
    import fused_ln as fused
except ImportError:
    logger.warning("fused-ln not found, run `python setup.py install` to build fused ln")
    fused = None


ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST = []

__all__ = [
    "Ernie35Model",
    "Ernie35PretrainedModel",
    "Ernie35ForCausalLM",
]


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, -np.inf, dtype=x.dtype)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def parallel_matmul(
    x,
    y,
    bias=None,
    transpose_y=False,
    tensor_parallel_degree=1,
    tensor_parallel_output=True,
    fuse_linear=False,
):

    if tensor_parallel_degree > 1 and y.is_distributed:
        pg = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        input_parallel = paddle.distributed.collective._c_identity(x, group=pg)
        if transpose_y:
            logits = paddle.matmul(input_parallel, y, transpose_y=True)
            if bias is not None:
                logits += bias
        else:
            if not fuse_linear:
                logits = F.linear(input_parallel, y, bias)
            else:
                logits = paddle.incubate.nn.functional.fused_linear(input_parallel, y, bias)  # hack for 逐位对齐

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=pg)

    else:
        logits = paddle.matmul(x, y, transpose_y=transpose_y)
        if bias is not None:
            logits += bias
        return logits


def finfo(dtype: paddle.dtype = None):
    if dtype is None:
        dtype = paddle.get_default_dtype()

    if dtype == paddle.bfloat16:
        # Numpy do not support `np.finfo(np.uint16)`, so try to construct a finfo object to fetch min value
        class BFloatFInfo:
            min = -3.3895313892515355e38

        return BFloatFInfo
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask.to("bool"), y, x)


def scaled_dot_product_attention(
    query_states, key_states, value_states, attention_mask, output_attentions, config, is_causal=True
):

    bsz, q_len, num_heads, _ = query_states.shape
    head_dim = config.hidden_size // config.num_attention_heads
    _, kv_seq_len, _, _ = value_states.shape

    if config.use_flash_attention and flash_attention is not None:
        # Flash Attention now ignore attention mask
        # Current Flash Attention doesn't support attn maskt
        # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
        # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]
        # without past keys
        attn_output, attn_weights = flash_attention(
            query_states,
            key_states,
            value_states,
            causal=is_causal and query_states.shape[1] != 1,
            return_softmax=output_attentions,
        )

        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return attn_output, attn_weights
    else:

        query_states = paddle.transpose(query_states, [0, 2, 1, 3]) / math.sqrt(head_dim)
        # merge with the next tranpose
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2]))

        if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )

        attn_weights = attention_mask + attn_weights
        attn_weights = paddle.maximum(
            attn_weights, paddle.to_tensor(float(finfo(query_states.dtype).min), dtype=query_states.dtype)
        )

        with paddle.amp.auto_cast(False):
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


def _make_causal_mask(input_ids_shape, past_key_values_length, dtype):
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape

    mask = paddle.full((target_length, target_length), float(finfo(dtype).min))

    mask_cond = paddle.arange(mask.shape[-1])
    mask = masked_fill(mask, mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0)

    if past_key_values_length > 0:
        mask = paddle.concat([paddle.zeros([target_length, past_key_values_length]), mask], axis=-1)

    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    if mask.ndim == 4:
        expanded_mask = mask
    elif mask.ndim == 3:
        expanded_mask = mask[:, None, :, :]
    else:
        batch_size, src_length = mask.shape[0], mask.shape[-1]
        tgt_length = tgt_length if tgt_length is not None else src_length

        expanded_mask = mask[:, None, None, :].expand([batch_size, 1, tgt_length, src_length])

    inverted_mask = 1.0 - expanded_mask
    return masked_fill(inverted_mask, inverted_mask.cast("bool"), float(finfo(dtype).min))


class LayerNorm(nn.LayerNorm):
    def __init__(self, config):
        super().__init__(config.hidden_size, epsilon=config.layer_norm_eps)


class FusedLayerNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.bias = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            is_bias=True,
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.variance_epsilon = config.layer_norm_eps

    def forward(self, hidden_states):
        return fused.fused_ln(hidden_states, self.weight, self.bias, self.variance_epsilon)[0]


class RotaryEmbedding(nn.Layer):
    def __init__(self, config, dim, max_position_embeddings=4096, base=10000):
        super().__init__()
        # dtype = paddle.get_default_dtype()
        self.config = config
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, dim, 2), dtype="float32") / dim))

        # higher acc using float32
        t = paddle.arange(max_position_embeddings, dtype="float32")
        freqs = paddle.einsum("i,j->ij", t, inv_freq.cast("float32"))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat([freqs, freqs], axis=-1)

        # [bs, seqlen, nhead, head_dim]
        self.cos_cached = emb.cos()  # [None, :, None, :]  # .astype(dtype)
        self.sin_cached = emb.sin()  # [None, :, None, :]  # .astype(dtype)

    def forward(self, x, seq_len=None, position_ids=None):
        if position_ids is not None:
            return self.cos_cached, self.sin_cached
        start = 0
        if self.config.enable_random_position_ids:
            if self.training:
                np_rng = np.random.RandomState(
                    int(os.getenv("TRAINER_GLOBAL_STEP", "0")) + (paddle.distributed.get_rank() * 100000)
                )
                pos_ids = np.array(list(np.sort(np_rng.permutation(self.max_position_embeddings)[:seq_len]))).astype(
                    "int64"
                )
                pos_ids = paddle.to_tensor(pos_ids)
            else:
                if seq_len <= 4096:
                    times = 8
                else:
                    times = self.max_position_embeddings // seq_len
                pos_ids = [times - 1]
                pos_ids += [times] * (seq_len - 1)
                pos_ids = paddle.cumsum(paddle.to_tensor(pos_ids))

            return (
                self.cos_cached[pos_ids],
                self.sin_cached[pos_ids],
            )

        return (
            self.cos_cached[start : start + seq_len, :],
            self.sin_cached[start : start + seq_len, :],
        )

    @classmethod
    def rotate_half(cls, x):
        """Rotates half the hidden dims of the input."""

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)

    @classmethod
    def apply_rotary_pos_emb(cls, q, k, cos, sin, offset: int = 0, position_ids=None):
        if position_ids is not None:
            assert offset == 0, offset
            cos = F.embedding(position_ids, cos)
            sin = F.embedding(position_ids, sin)
        else:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        cos = cos[:, offset : q.shape[1] + offset, None, :]
        sin = sin[:, offset : q.shape[1] + offset, None, :]

        # q_embed = (q * cos) + (rotate_half(q) * sin)
        # k_embed = (k * cos) + (rotate_half(k) * sin)

        cos = paddle.cast(cos, q.dtype)
        sin = paddle.cast(sin, q.dtype)
        q_embed = paddle.add(paddle.multiply(q, cos), paddle.multiply(cls.rotate_half(q), sin))
        k_embed = paddle.add(paddle.multiply(k, cos), paddle.multiply(cls.rotate_half(k), sin))
        return q_embed, k_embed


class Ernie35MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if config.tensor_parallel_degree > 1:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=config.use_bias,
                fuse_matmul_bias=config.fuse_linear,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=config.use_bias,
                fuse_matmul_bias=config.fuse_linear,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=config.use_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=config.use_bias)

        if config.tensor_parallel_degree > 1:
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=config.use_bias,
                fuse_matmul_bias=config.fuse_linear,
            )
        else:
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=config.use_bias)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x)


def rope_attn(
    mix_layer,
    query_states,
    key_states,
    value_states,
    attention_mask,
    position_ids,
    output_attentions=False,
    past_key_value=None,
    use_cache=False,
    rotary_emb=None,
    config=None,
):

    if mix_layer is not None:
        query_states, key_states, value_states = paddle.split(mix_layer, 3, axis=-1)

    kv_seq_len = key_states.shape[-3]
    offset = 0
    if past_key_value is not None:
        offset = past_key_value[0].shape[-3]
        kv_seq_len += offset

    cos, sin = rotary_emb(value_states, seq_len=kv_seq_len, position_ids=position_ids)

    query_states, key_states = rotary_emb.apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        position_ids=position_ids,
        offset=offset if position_ids is None else 0,
    )

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = paddle.concat([past_key_value[0], key_states], axis=1)
        value_states = paddle.concat([past_key_value[1], value_states], axis=1)

    past_key_value = (key_states, value_states) if use_cache else None
    attn_output, attn_weights = scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        output_attentions,
        config=config,
    )
    return attn_output, attn_weights, past_key_value


class Ernie35Attention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.use_recompute_attn = config.use_recompute_attn

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), "num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

        if config.tensor_parallel_degree > 1:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=config.use_bias,
                gather_output=False,
                fuse_matmul_bias=config.fuse_linear,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=config.use_bias,
                gather_output=False,
                fuse_matmul_bias=config.fuse_linear,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=config.use_bias,
                gather_output=False,
                fuse_matmul_bias=config.fuse_linear,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=config.use_bias,
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=config.use_bias,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=config.use_bias,
            )

        if config.tensor_parallel_degree > 1:
            self.o_proj = RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=config.use_bias,
                input_is_parallel=True,
                fuse_matmul_bias=config.fuse_linear,
            )
        else:
            self.o_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=config.use_bias,
            )

        self.rotary_emb = RotaryEmbedding(
            config,
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
        )
        self._cast_to_low_precison = False

        self.config = config

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.shape
        query_states = key_states = value_states = mix_layer = None
        # if self.fuse_attn:
        #     mix_layer = self.qkv_proj(hidden_states).reshape([bsz, q_len, self.num_heads, 3 * self.head_dim])
        # else:
        query_states = self.q_proj(hidden_states).reshape(shape=[bsz, q_len, self.num_heads, self.head_dim])
        key_states = self.k_proj(hidden_states).reshape(shape=[bsz, q_len, self.num_heads, self.head_dim])
        value_states = self.v_proj(hidden_states).reshape(shape=[bsz, q_len, self.num_heads, self.head_dim])

        _rope_attn = partial(rope_attn, rotary_emb=self.rotary_emb, config=self.config)
        if self.use_recompute_attn:
            assert past_key_value is None, "do not use kv cache in recompute"
            assert not use_cache
            attn_output, attn_weights, past_key_value = recompute(
                _rope_attn,
                mix_layer,
                query_states,
                key_states,
                value_states,
                attention_mask,
                position_ids,
                output_attentions,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            attn_output, attn_weights, past_key_value = _rope_attn(
                mix_layer,
                query_states,
                key_states,
                value_states,
                attention_mask,
                position_ids,
                output_attentions,
                past_key_value,
                use_cache=use_cache,
            )
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Ernie35MLPAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = config.intermediate_size

        self.use_recompute_attn = config.use_recompute_attn

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), "num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree
            self.intermediate_size_this_rank = self.intermediate_size // config.tensor_parallel_degree
        else:
            self.intermediate_size_this_rank = self.intermediate_size

        if config.tensor_parallel_degree > 1:
            self.qkv_gate_up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size * 3 + self.intermediate_size * 2,
                has_bias=config.use_bias,
                gather_output=False,
                fuse_matmul_bias=config.fuse_linear,
            )
            self.o_proj = RowParallelLinear(
                self.hidden_size + self.intermediate_size,
                self.hidden_size,
                has_bias=config.use_bias,
                input_is_parallel=True,
                fuse_matmul_bias=config.fuse_linear,
            )
        else:
            self.qkv_gate_up_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size * 3 + self.intermediate_size * 2,
                bias_attr=config.use_bias,
            )
            self.o_proj = nn.Linear(
                self.hidden_size + self.intermediate_size,
                self.hidden_size,
                bias_attr=config.use_bias,
            )

        self.rotary_emb = RotaryEmbedding(
            config,
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
        )
        self._cast_to_low_precison = False

        self.config = config

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        query_states = key_states = value_states = mix_layer = None

        mix_layer = self.qkv_gate_up_proj(hidden_states)
        mix_layer, up_states, gate_states = mix_layer.split(
            [self.head_dim * self.num_heads * 3, self.intermediate_size_this_rank, self.intermediate_size_this_rank],
            axis=-1,
        )
        mix_layer = mix_layer.reshape([bsz, q_len, self.num_heads, 3 * self.head_dim])
        _rope_attn = partial(rope_attn, rotary_emb=self.rotary_emb, config=self.config)

        if self.use_recompute_attn:
            assert past_key_value is None, "do not use kv cache in recompute"
            attn_output, attn_weights, past_key_value = recompute(
                _rope_attn,
                mix_layer,
                query_states,
                key_states,
                value_states,
                attention_mask,
                position_ids,
                output_attentions,
                use_reentrant=False,
            )
        else:
            attn_output, attn_weights, past_key_value = _rope_attn(
                mix_layer,
                query_states,
                key_states,
                value_states,
                attention_mask,
                position_ids,
                output_attentions,
                past_key_value,
                use_cache=use_cache,
            )

        ffn_output = F.silu(up_states) * gate_states

        output_states = paddle.concat([ffn_output, attn_output], axis=-1)
        output_states = self.o_proj(output_states)

        if not output_attentions:
            attn_weights = None

        return output_states, attn_weights, past_key_value


class Ernie35DecoderLayer(nn.Layer):
    def __init__(self, config, has_ffn=True, has_mha=True, parallel_attn_ffn=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn_ffn_no_mha = not parallel_attn_ffn and not has_mha

        Norm = LayerNorm
        if config.fuse_ln:
            Norm = FusedLayerNorm

        self.self_attn_mlp = self.self_attn = self.mlp = None
        if parallel_attn_ffn:
            logger.info("using parallel-attn")
            self.self_attn_mlp = Ernie35MLPAttention(config)
            self.input_layernorm = Norm(config)
            self.residual_add1 = FusedDropoutAdd(0.0, mode="upscale_in_train")
        else:
            logger.info("using normal-attn")
            if has_mha:
                self.self_attn = Ernie35Attention(config)
                self.residual_add1 = FusedDropoutAdd(0.0, mode="upscale_in_train")
                self.input_layernorm = Norm(config)

            if has_ffn:
                self.mlp = Ernie35MLP(config)
                self.post_attention_layernorm = Norm(config)
                self.residual_add2 = FusedDropoutAdd(0.0, mode="upscale_in_train")

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
            cache (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """
        if self.self_attn_mlp is not None:

            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn_mlp(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = self.residual_add1(hidden_states, residual)
        else:
            if self.self_attn is not None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)

                # Self Attention
                hidden_states, self_attn_weights, present_key_value = self.self_attn(
                    hidden_states=hidden_states,
                    past_key_value=past_key_value,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = self.residual_add1(hidden_states, residual)

            if self.mlp is not None:
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = self.residual_add2(hidden_states, residual)

        if self.attn_ffn_no_mha:
            present_key_value = None

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Ernie35PretrainedModel(PretrainedModel):
    config_class = Ernie35Config
    base_model_prefix = "ernie"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from conversion_utils import (
            o_proj_merge_fn,
            o_proj_split_fn,
            qkv_gate_up_proj_merge_fn,
            qkv_gate_up_proj_split_fn,
        )

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )
        qkv_gate_up_proj_fn = qkv_gate_up_proj_split_fn if is_split else qkv_gate_up_proj_merge_fn
        fuse_qkvgu_fn = qkv_gate_up_proj_fn(  # is_column: True
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_attention_heads,
        )
        o_proj_fn = o_proj_split_fn if is_split else o_proj_merge_fn
        fuse_o_proj_fn = o_proj_fn(  # is_column: False
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

        def get_tensor_parallel_split_mappings(num_layers, parallel_attn_hatf):
            final_actions = {}
            base_actions = {
                # Column Linear
                "layers.0.self_attn_mlp.qkv_gate_up_proj.weight": fuse_qkvgu_fn,
                "lm_head.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn_mlp.o_proj.weight": fuse_o_proj_fn,
            }
            if config.use_bias:
                base_actions.update(
                    {
                        # Column Linear
                        "layers.0.self_attn_mlp.qkv_gate_up_proj.bias": fuse_qkvgu_fn,
                        "lm_head.bias": partial(fn, is_column=True),
                    }
                )

            start = 0 if not parallel_attn_hatf else 1
            end = num_layers if not parallel_attn_hatf else num_layers - 1
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(start, end):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                if "layers.0." not in key:
                    final_actions[key] = action

            if parallel_attn_hatf:
                # Layer 0
                final_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
                final_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                final_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)
                final_actions["layers.0.self_attn.o_proj.weight"] = partial(fn, is_column=False)
                if config.use_bias:
                    final_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
                    final_actions["layers.0.self_attn.k_proj.bias"] = partial(fn, is_column=True)
                    final_actions["layers.0.self_attn.v_proj.bias"] = partial(fn, is_column=True)
                # Layer num_layers - 1
                final_actions[f"layers.{num_layers - 1}.mlp.gate_proj.weight"] = partial(fn, is_column=True)
                final_actions[f"layers.{num_layers - 1}.mlp.up_proj.weight"] = partial(fn, is_column=True)
                final_actions[f"layers.{num_layers - 1}.mlp.down_proj.weight"] = partial(fn, is_column=False)
                if config.use_bias:
                    final_actions[f"layers.{num_layers - 1}.mlp.gate_proj.bias"] = partial(fn, is_column=True)
                    final_actions[f"layers.{num_layers - 1}.mlp.up_proj.bias"] = partial(fn, is_column=True)

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers, config.parallel_attn_hatf)

        return mappings

    def _init_weights(self, layer):
        """Initialization hook"""
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state
        else:
            rng_tracker = contextlib.nullcontext

        if isinstance(
            layer,
            (
                ColumnParallelLinear,
                RowParallelLinear,
                VocabParallelEmbedding,
                Ernie35LMHead,
                nn.Embedding,
                nn.Linear,
            ),
        ):

            with rng_tracker():
                dtype = paddle.get_default_dtype()
                # layer.weight.set_value(
                #     paddle.randn(layer.weight.shape, dtype=dtype).scale(self.config.initializer_range)
                # )
                tmp = paddle.randn(layer.weight.shape, dtype=dtype).scale(self.config.initializer_range)
                src_tensor = tmp.value().get_tensor()
                layer.weight.value().get_tensor()._share_data_with(src_tensor)
                # layer.weight.copy_(tmp, True)

                logger.info(
                    f'dist-init-fc: shape={layer.weight.shape}, range={self.config.initializer_range},type={type(layer)},norm={layer.weight.astype("float32").norm().item()}'
                )

        elif isinstance(layer, RotaryEmbedding):
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            inv_freq = 1.0 / (layer.base ** (np.arange(0, head_dim, 2).astype("float32") / head_dim))
            # higher acc using float32
            t = np.arange(layer.max_position_embeddings, dtype="float32")
            freqs = np.einsum("i,j->ij", t, inv_freq)
            emb = np.concatenate([freqs, freqs], axis=-1)
            # [bs, seqlen, nhead, head_dim]
            cos_cached = np.cos(emb)[:, :]
            sin_cached = np.sin(emb)[:, :]
            layer.cos_cached.set_value(cos_cached)
            layer.sin_cached.set_value(sin_cached)


@register_base_model
class Ernie35Model(Ernie35PretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Ernie35DecoderLayer`]
    Args:
        config: Ernie35Config
    """

    def __init__(self, config: Ernie35Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        if config.tensor_parallel_degree > 1:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList()
        for i in range(config.num_hidden_layers):
            if not config.parallel_attn_hatf:
                _layer = Ernie35DecoderLayer(config, has_ffn=False, has_mha=False, parallel_attn_ffn=True)
            else:
                # no ffn in fisrt layer
                # no mha in last layer
                # Head mhA tail Ffn
                if i == 0:
                    _layer = Ernie35DecoderLayer(config, has_ffn=False, has_mha=True, parallel_attn_ffn=False)
                elif i == (config.num_hidden_layers - 1):
                    _layer = Ernie35DecoderLayer(config, has_ffn=True, has_mha=False, parallel_attn_ffn=False)
                else:
                    _layer = Ernie35DecoderLayer(config, has_ffn=False, has_mha=False, parallel_attn_ffn=True)

            self.layers.append(_layer)
            logger.info(
                f"building layer:{i}, mha={_layer.self_attn is not None},ffn={_layer.mlp is not None},paral-mha-ffn={_layer.self_attn_mlp is not None}"
            )

        Norm = LayerNorm
        if config.fuse_ln:
            Norm = FusedLayerNorm
        self.norm = Norm(config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @classmethod
    def _prepare_decoder_attention_mask(cls, attention_mask, input_shape, past_key_values_length, dtype):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length, dtype=dtype
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_length=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        combined_attention_mask = paddle.maximum(
            combined_attention_mask.astype(dtype), paddle.to_tensor(float(finfo(dtype).min), dtype=dtype)
        )
        return combined_attention_mask

    @paddle.jit.not_to_static
    def recompute_training(self, layer_module, hidden_states, attention_mask, position_ids, output_attentions):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_attentions, None)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            use_reentrant=False,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
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
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        # embed positions
        if attention_mask is None:
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
        )
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            has_gradient = not hidden_states.stop_gradient
            if self.config.use_recompute and has_gradient:
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
        )


class Ernie35PretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for Ernie35.
    It calculates the final loss.
    """

    def __init__(self, config, return_tuple=True):
        super(Ernie35PretrainingCriterion, self).__init__()
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.config = config
        self.return_tuple = return_tuple
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output

        if self.enable_parallel_cross_entropy:
            logger.info("using parallel cross entroy, take care")
            self.loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )

    def forward(self, prediction_scores, masked_lm_labels):
        if self.enable_parallel_cross_entropy:
            assert (
                prediction_scores.shape[-1] != self.config.vocab_size
            ), f"enable_parallel_cross_entropy, the vocab_size should be splited: {prediction_scores.shape[-1]}, {self.config.vocab_size}"

        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2))
            lossmask = masked_lm_labels != self.ignored_index
            if (~lossmask).all():
                loss = paddle.mean(masked_lm_loss) * 0.0
                loss_sum = masked_lm_loss.sum().detach()
            else:
                lossmask = lossmask.reshape([-1]).cast(paddle.float32)
                masked_lm_loss = paddle.sum(masked_lm_loss.cast(paddle.float32).reshape([-1]) * lossmask)
                loss = masked_lm_loss / lossmask.sum()
                loss_sum = masked_lm_loss.sum().detach()
        if not self.return_tuple:
            if self.training:
                return loss
            return loss_sum
        return loss, loss_sum


class Ernie35LMHead(nn.Layer):
    def __init__(self, config):
        super(Ernie35LMHead, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        if vocab_size != config.vocab_size:
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[vocab_size, config.hidden_size]
                    if config.tie_word_embeddings
                    else [config.hidden_size, vocab_size],
                    dtype=paddle.get_default_dtype(),
                )
                if config.weight_share_add_bias and config.use_bias:
                    self.bias = self.create_parameter(
                        shape=[vocab_size],
                        dtype=paddle.get_default_dtype(),
                    )
                else:
                    self.bias = None
        else:
            self.weight = self.create_parameter(
                shape=[vocab_size, config.hidden_size]
                if config.tie_word_embeddings
                else [config.hidden_size, vocab_size],
                dtype=paddle.get_default_dtype(),
            )
            if config.weight_share_add_bias and config.use_bias:
                self.bias = self.create_parameter(
                    shape=[vocab_size],
                    dtype=paddle.get_default_dtype(),
                )
            else:
                self.bias = None

        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = True if (vocab_size != config.vocab_size) else False
        if config.weight_share_add_bias and config.use_bias:
            self.bias.is_distributed = True if (vocab_size != config.vocab_size) else False

        if self.weight.is_distributed:
            self.weight.split_axis = 1
        if config.weight_share_add_bias and config.use_bias and self.bias.is_distributed:
            self.bias.split_axis = 0

    def forward(self, hidden_states, tensor_parallel_output=None):
        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output
        logits = parallel_matmul(
            hidden_states,
            self.weight,
            bias=self.bias,
            transpose_y=self.config.tie_word_embeddings,
            tensor_parallel_degree=self.config.tensor_parallel_degree,
            tensor_parallel_output=tensor_parallel_output,
            fuse_linear=self.config.fuse_linear,
        )
        return logits


class Ernie35ForCausalLM(Ernie35PretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # initialize-trick for big model, see https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md#std-init
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"change initializer-range from {config.initializer_range} to {new_initializer_range}")
        config.initializer_range = new_initializer_range
        self.config = config

        self.ernie = Ernie35Model(config)
        self.lm_head = Ernie35LMHead(config)
        self.criterion = Ernie35PretrainingCriterion(config)

        self.tie_weights()  # maybe weight share

    # Initialize weights and apply final processing
    def _post_init(self, original_init, *args, **kwargs):
        super()._post_init(self, original_init, *args, **kwargs)
        factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
        logger.info(f"using post init div: factor:{factor}")
        with paddle.no_grad():
            for l in self.ernie.layers:
                if l.self_attn_mlp is not None:
                    l.self_attn_mlp.o_proj.weight.scale_(factor)
                if l.self_attn is not None:
                    l.self_attn.o_proj.weight.scale_(factor)
                if l.mlp is not None:
                    l.mlp.down_proj.weight.scale_(factor)

    def get_input_embeddings(self):
        return self.ernie.embed_tokens

    def set_input_embeddings(self, value):
        self.ernie.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.ernie = decoder

    def get_decoder(self):
        return self.ernie

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id
        ).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("int64")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")
        return attention_mask

    def prepare_inputs_for_generation(
        self, input_ids, use_cache=False, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,  # use_cache,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "return_dict": True,
            }
        )
        return model_inputs

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        interval = int(os.getenv("pos_decoding_interval", "2"))
        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[:, -1:] + interval], axis=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype="int64")], axis=-1
                )
        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat([role_ids, role_ids[:, -1:]], axis=-1)

        return model_kwargs

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        ignored_index=0,  # no use
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        def progressive_seq(x, y):
            globel_step = int(os.getenv("TRAINER_GLOBAL_STEP", "0"))
            if globel_step < 500:
                return x[:, :512], y[:, :512]
            if globel_step < 1000:
                return x[:, :1024], y[:, :1024]
            if globel_step < 1500:
                return x[:, :2048], y[:, :2048]
            return x, y

        if self.config.use_progressive_seq_len and self.training:
            input_ids, labels = progressive_seq(input_ids, labels)

        outputs = self.ernie(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is togather with ParallelCrossEntropy
        # tensor_parallel_output = (
        #     self.config.tensor_parallel_output and labels is not None and self.config.tensor_parallel_degree > 1
        # )

        logits = self.lm_head(
            hidden_states,
        )  # tensor_parallel_output=tensor_parallel_output)

        if return_dict:
            if labels is not None:
                loss, _ = self.criterion(logits, labels)
            else:
                loss = None
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        assert labels is not None
        loss, loss_sum = self.criterion(logits, labels)
        return loss, loss_sum
