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
from typing import Any, Dict, List, Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .. import PretrainedModel, register_base_model
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPast,
    ModelOutput,
)
from .configuration import ChatGLMv2Config

CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm2-6b",
    # See all ChatGLM models at https://huggingface.co/models?filter=chatglm
]


def baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0):
    return beta * input + alpha * paddle.matmul(batch1, batch2)


def split_tensor_along_last_dim(
    tensor: paddle.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[paddle.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    # last_dim = tensor.dim() - 1
    # last_dim_size = tensor.shape[last_dim] // num_partitions
    # Split.
    tensor_list = paddle.split(tensor, num_partitions, axis=-1)
    # # Note: paddle.split does not create contiguous tensors by default.
    # if contiguous_split_chunks:
    #     return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, original_impl=False):
        super().__init__()
        self.dtype = paddle.get_default_dtype()
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
        idx_theta = paddle.outer(seq_idx, theta).astype(self.dtype)

        cache = paddle.stack([paddle.cos(idx_theta), paddle.sin(idx_theta)], axis=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if self.dtype in (paddle.float16, paddle.bfloat16, paddle.int8):
            cache = cache.astype(self.dtype)
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

        self.dtype = paddle.get_default_dtype()
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # Raw attention scores
        # [b, np, sq, sk]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape([output_size[2], output_size[0] * output_size[1], -1])
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.reshape([output_size[3], output_size[0] * output_size[1], -1])

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = paddle.empty(
            [output_size[0] * output_size[1], output_size[2], output_size[3]],
            dtype=query_layer.dtype,
        )

        # Raw attention scores. [b * np, sq, sk]
        # matmul_result = paddle.bmm(
        #     query_layer.transpose([1, 0, 2]),
        #     key_layer.transpose([1, 2, 0])
        #     ) * (1.0 / self.norm_factor)
        matmul_result = baddbmm(
            matmul_input_buffer,
            query_layer.transpose([1, 0, 2]),  # [b * np, sq, hn]
            key_layer.transpose([1, 2, 0]),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.reshape(output_size)
        # print("attention_scores", attention_scores[..., :3].numpy().tolist())

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.astype("float32")
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = paddle.tril(
                paddle.ones((output_size[0], 1, output_size[2], output_size[3]), dtype="bool")
            )
            attention_mask = ~attention_mask

        if attention_mask is not None:
            attention_scores = paddle.where(
                attention_mask > 0, paddle.full_like(attention_scores, -1e6), attention_scores
            )

        attention_probs = F.softmax(attention_scores.astype("float32"), axis=-1)
        attention_probs = attention_probs.astype(self.dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

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

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(
            config.hidden_size,
            self.qkv_hidden_size,
            bias_attr=config.add_bias_linear or config.add_qkv_bias,
        )

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias_attr=config.add_bias_linear)

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return paddle.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True):
        # hidden_states: [seq_length, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [seq_length, b, h] --> [seq_length, b, (np * 3 * hn)]

        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                axis=-1,
            )
            query_layer = query_layer.reshape(
                query_layer.shape[:-1] + [self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
            )
            key_layer = key_layer.reshape(
                key_layer.shape[:-1] + [self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head]
            )
            value_layer = value_layer.reshape(
                value_layer.shape[:-1]
                + [self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head]
            )
        else:
            new_tensor_shape = mixed_x_layer.shape[:-1] + [
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            ]
            mixed_x_layer = mixed_x_layer.reshape(new_tensor_shape)

            # [seq_length, b, np, 3 * hn] --> 3 [seq_length, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

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

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.tile(
                [1, 1, 1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, 1]
            )
            key_layer = key_layer.reshape(
                key_layer.shape[:2] + [self.num_attention_heads_per_partition, self.hidden_size_per_attention_head]
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.tile(
                [1, 1, 1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, 1]
            )
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

    def __init__(self, config: ChatGLMv2Config, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size * 2, bias_attr=self.add_bias)

        def swiglu(x):
            x = paddle.chunk(x, 2, axis=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias_attr=self.add_bias,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
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
        for index in range(self.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)

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
    # base_model_prefix = "transformer"
    base_model_prefix = "chatglm_v2"

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = paddle.tril(paddle.ones([batch_size, seq_length, seq_length]))
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = paddle.concat(
                [paddle.ones([batch_size, seq_length, past_length]), full_attention_mask], axis=-1
            )
        if padding_mask is not None:
            print(padding_mask.shape, full_attention_mask.shape)
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).astype("bool")
        full_attention_mask.unsqueeze(1)
        return full_attention_mask

    def get_position_ids(self, input_ids):
        batch_size, seq_length = input_ids.shape
        position_ids = paddle.arange(seq_length, dtype="int64").unsqueeze(0).tile([batch_size, 1])
        return position_ids


class Embedding(nn.Layer):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMv2Config):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
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
        # output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.astype("bool").all()) or (
                past_key_values and seq_length != 1
            ):
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
            # attentions=all_self_attentions,
        )


class ChatGLMv2ForConditionalGeneration(ChatGLMv2PretrainedModel):
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
            lm_logits = lm_logits.astype("float32")

            # Shift so that tokens < n predict n and flatten the logits and labels
            shift_logits = lm_logits[..., :-1, :]
            shift_logits = shift_logits.reshape([-1, shift_logits.shape[-1]])
            shift_labels = labels[..., 1:].reshape([-1])

            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

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
