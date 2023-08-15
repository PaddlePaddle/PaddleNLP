# Copyright (c) 2023 ChatGLM2-6B Model Team and PaddlePaddle Authors. All Rights Reserved.
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

import math
from functools import partial
from typing import Any, Dict, Optional, Tuple

import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute

from .. import PretrainedModel, register_base_model
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPast,
    ModelOutput,
)
from .configuration import CHATGLM_V2_PRETRAINED_RESOURCE_FILES_MAP, ChatGLMv2Config

__all__ = [
    "ChatGLMv2Model",
    "ChatGLMv2PretrainedModel",
    "ChatGLMv2ForCausalLM",
]


def assign_kv_heads(num_kv_heads, num_gpus):
    # Initialize the assignment list
    """
    Assign kv heads to different GPUs in the Tensor Parallel Setup

    Examples:
        assign_kv_heads(num_kv_heads=1, num_gpus=2): [[0], [0]]
        assign_kv_heads(num_kv_heads=2, num_gpus=2): [[0], [1]]
        assign_kv_heads(num_kv_heads=4, num_gpus=2): [[0,1], [2,3]]
        assign_kv_heads(num_kv_heads=1, num_gpus=4): [[0],[0],[0],[0]]
        assign_kv_heads(num_kv_heads=2, num_gpus=4): [[0],[0],[1],[1]]
        assign_kv_heads(num_kv_heads=4, num_gpus=4): [[0],[1],[2],[3]]
    """
    assignment_list = [[] for _ in range(num_gpus)]
    # Case 1: more heads than cards
    if num_kv_heads > num_gpus:
        num_heads_per_card = num_kv_heads // num_gpus
        for i in range(num_gpus):
            for j in range(num_heads_per_card):
                assignment_list[i].append(i * num_heads_per_card + j)
    # Case 2: more cards than heads. each card get only 1 head.
    else:
        num_card_per_heads = num_gpus // num_kv_heads
        for i in range(num_kv_heads):
            for j in range(num_card_per_heads):
                assignment_list[i * num_card_per_heads + j].append(i)
    return assignment_list


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, original_impl=False):
        super().__init__()
        self.default_dtype = paddle.get_default_dtype()
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(self, seq_len: int, n_elem: int, base: int = 10000):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (paddle.arange(0, n_elem, 2, dtype="float32") / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = paddle.arange(0, seq_len, dtype=theta.dtype)

        # Calculate the product of position index and $\theta_i$
        idx_theta = paddle.outer(seq_idx, theta).astype(self.default_dtype)

        cache = paddle.stack([paddle.cos(idx_theta), paddle.sin(idx_theta)], axis=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if self.default_dtype in (paddle.float16, paddle.bfloat16, paddle.int8):
            cache = cache.astype(self.default_dtype)
            # cache = cache.bfloat16() if dtype == paddle.bfloat16 else cache.astype("float16")
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(seq_len=max_seq_len, n_elem=self.dim)


# @paddle.jit.script
def apply_rotary_pos_emb(x: paddle.Tensor, rope_cache: paddle.Tensor) -> paddle.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape([sq, -1, np, rot_dim // 2, 2])
    rope_cache = rope_cache.reshape([sq, -1, 1, xshaped.shape[3], 2])
    x_out2 = paddle.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return paddle.concat((x_out2, x_pass), axis=-1)


class RMSNorm(nn.Layer):
    def __init__(self, hidden_size, epsilon=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.epsilon = 1e-5 if epsilon is None else epsilon

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
        hidden_states = paddle.rsqrt(variance + self.epsilon) * hidden_states
        output = (hidden_states * self.weight).astype(input_dtype)

        # if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
        #     hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return output


class CoreAttention(nn.Layer):
    def __init__(self, config: ChatGLMv2Config, layer_number):
        super(CoreAttention, self).__init__()

        self.default_dtype = paddle.get_default_dtype()
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        if config.tensor_parallel_degree > 1:
            self.num_attention_heads_per_partition = config.num_attention_heads // config.tensor_parallel_degree
        else:
            self.num_attention_heads_per_partition = config.num_attention_heads
        self.hidden_size_per_partition = config.kv_channels * self.num_attention_heads_per_partition
        self.hidden_size_per_attention_head = self.hidden_size_per_partition // self.num_attention_heads_per_partition

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # Raw attention scores
        # [batch_size, num_heads, query_length, key_length]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        # [query_length, batch_size, num_heads, hidden] -> [query_length, batch_size * num_heads, hidden]
        query_layer = query_layer.reshape([output_size[2], output_size[0] * output_size[1], -1])
        # [key_length, batch_size, num_heads, hidden] -> [key_length, batch_size * num_heads, hidden]
        key_layer = key_layer.reshape([output_size[3], output_size[0] * output_size[1], -1])

        # Raw attention scores. [batch_size * num_heads, query_length, key_length]
        matmul_result = paddle.bmm(query_layer.transpose([1, 0, 2]), key_layer.transpose([1, 2, 0])) * (
            1.0 / self.norm_factor
        )

        # change view to [batch_size, num_heads, query_length, key_length]
        attention_scores = matmul_result.reshape(output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [batch_size, num_heads, query_length, key_length]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.astype("float32")
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff

        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores.astype("float32"), axis=-1)
        attention_probs = attention_probs.astype(self.default_dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)
        # [batch_size, num_heads, query_length, key_length]

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape([value_layer.shape[0], output_size[0] * output_size[1], -1])
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.reshape([output_size[0] * output_size[1], output_size[2], -1])
        # matmul: [b * np, sq, hn]
        context_layer = paddle.bmm(attention_probs, value_layer.transpose([1, 0, 2]))
        # change view [b, np, sq, hn]
        context_layer = context_layer.reshape(output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.transpose([2, 0, 1, 3])
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context_layer.shape[:-2] + [self.hidden_size_per_partition]
        context_layer = context_layer.reshape(new_context_shape)

        return context_layer


class SelfAttention(nn.Layer):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMv2Config, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        assert (
            config.kv_channels * config.num_attention_heads == config.hidden_size
        ), "`kv_channels` * `num_attention_heads` must equal to `hidden_size`"

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads
        self.core_attention = CoreAttention(config, self.layer_number)
        self.num_multi_query_groups_per_partition = config.multi_query_group_num

        if config.tensor_parallel_degree > 1:
            self.query = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size,
                config.hidden_size,
                has_bias=config.add_bias_linear or config.add_qkv_bias,
                gather_output=False,
            )
            self.dense = fleet.meta_parallel.RowParallelLinear(
                config.hidden_size, config.hidden_size, input_is_parallel=True, has_bias=config.add_bias_linear
            )
            self.num_attention_heads_per_partition = config.num_attention_heads // config.tensor_parallel_degree
            self.kv_indices = paddle.to_tensor(
                assign_kv_heads(config.multi_query_group_num, config.tensor_parallel_degree)[
                    config.tensor_parallel_rank
                ]
            )
        else:
            self.query = nn.Linear(
                config.hidden_size,
                config.hidden_size,
                bias_attr=config.add_bias_linear or config.add_qkv_bias,
            )
            self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=config.add_bias_linear)
            self.num_attention_heads_per_partition = config.num_attention_heads
            self.kv_indices = None

        self.key = nn.Linear(
            config.hidden_size,
            self.hidden_size_per_attention_head * config.multi_query_group_num,
            bias_attr=config.add_qkv_bias,
        )
        self.value = nn.Linear(
            config.hidden_size,
            self.hidden_size_per_attention_head * config.multi_query_group_num,
            bias_attr=config.add_qkv_bias,
        )

    def forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
        seq_length, batch_size, hidden_size = hidden_states.shape
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = query_layer.reshape(
            [seq_length, batch_size, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
        )
        key_layer = key_layer.reshape([seq_length, batch_size, -1, self.hidden_size_per_attention_head])
        value_layer = value_layer.reshape([seq_length, batch_size, -1, self.hidden_size_per_attention_head])

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if use_cache:
            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                key_layer = paddle.concat((cache_k, key_layer), axis=0)
                value_layer = paddle.concat((cache_v, value_layer), axis=0)
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.kv_indices is not None:
            key_layer = paddle.index_select(key_layer, self.kv_indices, axis=2)
            value_layer = paddle.index_select(value_layer, self.kv_indices, axis=2)
            multiplier = self.num_attention_heads_per_partition // self.kv_indices.shape[0]
        else:
            multiplier = self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition

        key_layer = key_layer.unsqueeze(-2).tile([1, 1, 1, multiplier, 1])
        key_layer = key_layer.reshape(
            key_layer.shape[:2] + [self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
        )
        value_layer = value_layer.unsqueeze(-2).tile([1, 1, 1, multiplier, 1])
        value_layer = value_layer.reshape(
            value_layer.shape[:2] + [self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
        )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [seq_length, b, h]
        # =================

        output = self.dense(context_layer)
        return output, kv_cache


class MLP(nn.Layer):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMv2Config):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        if config.tensor_parallel_degree > 1:
            self.dense_h_to_4h = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size, config.ffn_hidden_size * 2, has_bias=self.add_bias, gather_output=False
            )
            self.dense_4h_to_h = fleet.meta_parallel.RowParallelLinear(
                config.ffn_hidden_size, config.hidden_size, input_is_parallel=True, has_bias=self.add_bias
            )
        else:
            # Project to 4h due to swiglu doubling the output width, see https://arxiv.org/pdf/2002.05202.pdf
            self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size * 2, bias_attr=self.add_bias)
            # Project back to h.
            self.dense_4h_to_h = nn.Linear(
                config.ffn_hidden_size,
                config.hidden_size,
                bias_attr=self.add_bias,
            )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # Special Slicing to accomodate Tensor Parallel
        # Even channels is ffc_fc, odd channels is gate
        ffn_fc = intermediate_parallel[..., 0::2]
        gate = intermediate_parallel[..., 1::2]
        intermediate_parallel = F.silu(ffn_fc) * gate
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Layer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMv2Config, layer_number):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else nn.LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, epsilon=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, epsilon=config.layernorm_epsilon)

        # MLP
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output, attention_mask, rotary_pos_emb, kv_cache=kv_cache, use_cache=use_cache
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = F.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = F.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output
        return output, kv_cache


class GLMTransformer(nn.Layer):
    """Transformer class."""

    def __init__(self, config: ChatGLMv2Config):
        super(GLMTransformer, self).__init__()
        self.enable_recompute = False
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_hidden_layers = config.num_hidden_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number)

        self.layers = nn.LayerList([build_layer(i + 1) for i in range(self.num_hidden_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else nn.LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, epsilon=config.layernorm_epsilon)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        rotary_embeds: paddle.Tensor,
        kv_cache: paddle.Tensor,
        use_cache: bool,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states, kv_cache = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            rotary_embeds,
            kv_cache,
            use_cache,
            use_reentrant=False,
        )
        return hidden_states, kv_cache

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_hidden_layers)]
        presents = () if use_cache else None
        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None

        zero = paddle.zeros(attention_mask.shape, dtype=hidden_states.dtype)
        neg_inf = paddle.full_like(attention_mask, paddle.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
        attention_mask = paddle.where(attention_mask, zero, neg_inf)

        for index in range(self.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)

            if self.enable_recompute and not hidden_states.stop_gradient:
                hidden_states, kv_cache = self.recompute_training(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache,
                )
            else:
                hidden_states, kv_cache = layer(
                    hidden_states, attention_mask, rotary_pos_emb, kv_cache=kv_caches[index], use_cache=use_cache
                )

            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMv2PretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = ChatGLMv2Config
    pretrained_resource_files_map = CHATGLM_V2_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "chatglm_v2"

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape

        # casual mask
        casual_mask = paddle.tril(paddle.ones([batch_size, 1, seq_length, seq_length])).astype("bool")
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            casual_mask = paddle.concat(
                [paddle.ones([batch_size, 1, seq_length, past_length], dtype="bool"), casual_mask], axis=-1
            )

        # seq_mask
        if padding_mask is None:
            padding_mask = paddle.ones((batch_size, 1, seq_length, seq_length + past_length), dtype="bool")
        if len(padding_mask.shape) == 2:
            # from Tokenizer
            padding_mask = (
                padding_mask.unsqueeze(axis=[1, 2])
                .expand([batch_size, 1, seq_length, seq_length + past_length])
                .astype("bool")
            )
        elif len(padding_mask.shape) == 3:
            # [batch_size,tgt_length, src_length] -> [batch_size, 1, tgt_length, src_length]
            padding_mask = padding_mask.unsqueeze(1).astype("bool")
        elif len(padding_mask.shape) == 4:
            padding_mask = padding_mask.astype("bool")

        casual_mask = casual_mask & padding_mask

        return casual_mask

    def get_position_ids(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = paddle.arange(seq_length, dtype="int64").unsqueeze(0).tile([batch_size, 1])
        return position_ids

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_hidden_layers):
            final_actions = {}
            base_actions = {
                # Column Linear
                "encoder.layers.0.mlp.dense_h_to_4h.weight": partial(fn, is_column=True),
                "encoder.layers.0.self_attention.query.bias": partial(fn, is_column=True),
                "encoder.layers.0.self_attention.query.weight": partial(fn, is_column=True),
                "output_layer.weight": partial(fn, is_column=True),
                # Row Linear
                "embedding.word_embeddings.weight": partial(fn, is_column=False),
                "encoder.layers.0.self_attention.dense.weight": partial(fn, is_column=False),
                "encoder.layers.0.mlp.dense_4h_to_h.weight": partial(fn, is_column=False),
            }
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_hidden_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings


class Embedding(nn.Layer):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMv2Config):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        if config.tensor_parallel_degree > 1:
            self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.word_embeddings = nn.Embedding(config.padded_vocab_size, self.hidden_size)
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        # Data format change to avoid explicit tranposes
        # [batch_size, seq_length, hidden_size] --> [seq_length, batch_size, hidden_size].
        embeddings = embeddings.transpose([1, 0, 2])
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.astype("float32")
        return embeddings


@register_base_model
class ChatGLMv2Model(ChatGLMv2PretrainedModel):
    def __init__(self, config: ChatGLMv2Config, empty_init=True):
        super().__init__(config)
        self.embedding = Embedding(config)

        # Rotary positional embeddings
        self.max_sequence_length = config.max_sequence_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2)
        self.encoder = GLMTransformer(config)
        if config.tensor_parallel_degree > 1:
            self.output_layer = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size,
                config.padded_vocab_size,
                has_bias=False,
                gather_output=not config.tensor_parallel_output,
            )
        else:
            self.output_layer = nn.Linear(config.hidden_size, config.padded_vocab_size, bias_attr=False)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    def forward(
        self,
        input_ids,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        full_attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor, paddle.Tensor], ...]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.max_sequence_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3])

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )


class ChatGLMv2ForCausalLM(ChatGLMv2PretrainedModel):
    def __init__(self, config: ChatGLMv2Config):
        super().__init__(config)
        self.max_sequence_length = config.max_sequence_length
        self.chatglm_v2 = ChatGLMv2Model(config)

    def update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs[1] if isinstance(outputs, tuple) else outputs["past_key_values"]

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            new_attention_mask = paddle.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
            model_kwargs["attention_mask"] = paddle.concat([attention_mask, new_attention_mask], axis=-1)

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = paddle.concat([position_ids, new_position_id], axis=-1)

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: paddle.Tensor,
        past_key_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        is_first_forward: bool = True,
        **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)
        if not is_first_forward:
            position_ids = position_ids[..., -1:]
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": True,
        }

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.chatglm_v2(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.chatglm_v2.output_layer(hidden_states)
        lm_logits = lm_logits.transpose([1, 0, 2])

        loss = None
        if labels is not None:
            reshaped_logits = lm_logits.reshape([-1, lm_logits.shape[-1]]).astype("float32")
            reshaped_labels = labels.reshape([-1])

            if self.config.tensor_parallel_degree > 1 and self.config.tensor_parallel_output:
                loss_fn = fleet.meta_parallel.ParallelCrossEntropy()
            else:
                loss_fn = nn.CrossEntropyLoss(reduction="none")

            loss_mask = (labels != -100).astype("float32")
            loss = loss_fn(reshaped_logits, reshaped_labels)
            loss = paddle.sum(loss.reshape([-1]).cast(paddle.float32) * loss_mask.reshape([-1]).cast(paddle.float32))
            loss = loss / loss_mask.sum()

            lm_logits = lm_logits.astype(hidden_states.dtype)
            loss = loss.astype(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
