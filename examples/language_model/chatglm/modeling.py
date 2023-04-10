# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""GLM model"""
from __future__ import annotations

import math
import re
from functools import partial
from typing import Any, Dict, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from configuration import CHATGLM_PRETRAINED_RESOURCE_FILES_MAP, ChatGLMConfig
from paddle import Tensor
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.transformers.model_outputs import CausalLMOutputWithPast, ModelOutput
from paddlenlp.utils.converter import StateDictNameMapping
from paddlenlp.utils.env import CONFIG_NAME
from paddlenlp.utils.log import logger

__all__ = [
    "ChatGLMModel",
    "ChatGLMPretrainedModel",
    "ChatGLMForConditionalGeneration",
]


class PrefixEncoder(nn.Layer):
    """
    The prefix encoder for P-Tuning v2.
    Input shape: [batch_size, prefix_length]
    Output shape: [batch_size, prefix_length, 2 * num_layers * hidden_size]
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.num_layers * config.hidden_size * 2),
            )
        else:
            self.embedding = nn.Embedding(config.pre_seq_len, config.num_layers * config.hidden_size * 2)

    def forward(self, prefix: paddle.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class RotaryEmbeddings(nn.Layer):
    def __init__(self, hidden_size, base=10000, dtype=paddle.float16, learnable=False):
        super().__init__()
        inv_freq = 1.0 / (base ** (paddle.arange(0, hidden_size, 2).astype("float32") / hidden_size))
        inv_freq = inv_freq.astype(dtype)
        self.learnable = learnable
        if learnable:
            self.inv_freq = nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer("inv_freq", inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.dtype = dtype

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        # x.shape = [b, s, n, h/n/2]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            # [s]
            t = paddle.arange(seq_len).astype(self.inv_freq.dtype)
            # [s, h/n/2]
            freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
            # [s, h/n]
            emb = paddle.concat([freqs, freqs], axis=-1)
            if self.dtype == paddle.float16:
                emb = emb.astype("float32")
            # [s, 1, h/n]
            cos_cached = emb.cos().unsqueeze(1)
            sin_cached = emb.sin().unsqueeze(1)

            if self.dtype == paddle.float16:
                cos_cached = cos_cached.astype(self.dtype)
                sin_cached = sin_cached.astype(self.dtype)

            if self.learnable:
                return cos_cached, sin_cached

            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)


class ChatGLMAttention(nn.Layer):
    """
    Self-attention layer performs multiple attention to jointly attending to
    information from different representation subspaces.
    """

    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.position_encoding_2d = config.position_encoding_2d
        self.rotary_embeddings = RotaryEmbeddings(
            self.hidden_size // (self.num_attention_heads * 2)
            if self.position_encoding_2d
            else self.hidden_size // self.num_attention_heads,
            base=10000,
            dtype=self.config.paddle_dtype,
            learnable=False,
        )
        self.scale_mask_softmax = None

        self.attention_scale = config.attention_scale

        if config.tensor_parallel_degree > 1:
            self.query_key_value = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size, 3 * config.hidden_size, has_bias=True, gather_output=False
            )
            self.dense = fleet.meta_parallel.RowParallelLinear(
                config.hidden_size, config.hidden_size, input_is_parallel=True, has_bias=True
            )
            self.num_attention_heads = config.num_attention_heads // config.tensor_parallel_degree
        else:
            self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
        # self.output_dropout = nn.Dropout(config.output_dropout_prob)

    def _rotate_half(self, x):
        x1, x2 = paddle.chunk(x, 2, axis=-1)
        return paddle.concat([-x2, x1], axis=-1)

    def _apply_rotary_position_embed_index(self, q, k, cos, sin, position_ids):
        # q.shape = [s, b, n, h/n/2], cos.shape = [s, 1, h/n], position_ids.shape = [s, b]
        # [s, b, 1, h/n]
        cos, sin = F.embedding(position_ids, cos.squeeze(1)).unsqueeze(2), F.embedding(
            position_ids, sin.squeeze(1)
        ).unsqueeze(2)
        # [s, b, n, h/n]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    def _core_attention(self, q_layer: Tensor, k_layer: Tensor, position_ids: Tensor):
        # Set store_true, position_encoding_2d=False by default.
        if self.config.position_encoding_2d:
            # [s, b, n, h/n/2]
            q1, q2 = paddle.chunk(q_layer, 2, axis=-1)
            k1, k2 = paddle.chunk(k_layer, 2, axis=-1)
            # [s, 1, h/n]
            cos, sin = self.rotary_embeddings(q1, seq_len=position_ids.max() + 1)
            # [s, b]
            position_ids, block_position_ids = position_ids[:, 0, :].transpose([1, 0]), position_ids[
                :, 1, :
            ].transpose([1, 0])

            # [s, b, n, h/n]
            q1, k1 = self._apply_rotary_position_embed_index(q1, k1, cos, sin, position_ids)
            q2, k2 = self._apply_rotary_position_embed_index(q2, k2, cos, sin, block_position_ids)
            q_layer = paddle.concat([q1, q2], axis=-1)
            k_layer = paddle.concat([k1, k2], axis=-1)
        else:
            # [s, b]
            position_ids = position_ids.transpose([1, 0])
            # [s, 1, h/n]
            cos, sin = self.rotary_embeddings(q_layer, seq_len=position_ids.max() + 1)
            # [s, b, n, h/n]
            q_layer, k_layer = self._apply_rotary_position_embed_index(q_layer, k_layer, cos, sin, position_ids)
        return q_layer, k_layer

    def forward(
        self,
        hidden_states: Tensor,
        ltor_mask: Tensor,
        position_ids: Tensor,
        use_cache: bool = False,
        cache: Tensor = None,
        layer_id=0,
    ):
        # [s, b, h]
        query_length, batch_size = hidden_states.shape[:2]
        # [s, b, 3h]
        mixed_layer = self.query_key_value(hidden_states)
        # [s, b, n, 3h//n]
        mixed_layer = mixed_layer.reshape(
            [query_length, batch_size, self.num_attention_heads, self.attention_head_size * 3]
        )
        # [s, b, n, h//n]
        q_layer, k_layer, v_layer = paddle.split(mixed_layer, 3, axis=-1)
        # [s, b, n, h/n]
        q_layer, k_layer = self._core_attention(q_layer, k_layer, position_ids)

        if cache is not None:
            cache_k, cache_v = cache[0], cache[1]
            # [s + c, b, n, h/n]
            k_layer = paddle.concat([cache_k, k_layer], axis=0)
            v_layer = paddle.concat([cache_v, v_layer], axis=0)

        seq_length, batch_size, num_heads, hidden_size = k_layer.shape

        cache_kv = None
        if use_cache:
            cache_kv = (k_layer, v_layer)

        attention_scale_coeff = float(layer_id) + 1.0
        if self.attention_scale:
            # [s, b, n, h/n]
            q_layer = q_layer / (math.sqrt(self.attention_head_size) * attention_scale_coeff)

        # [b, n, s, s]
        output_shape = [q_layer.shape[1], q_layer.shape[2], q_layer.shape[0], k_layer.shape[0]]

        # [s, b * n, h/n]
        q_layer = q_layer.reshape([output_shape[2], output_shape[0] * output_shape[1], -1])
        k_layer = k_layer.reshape([output_shape[3], output_shape[0] * output_shape[1], -1])

        # [b * n , s, s] = matmul([b * n, s, h/n],  [b * n, h/n, s])
        attention_scores = paddle.matmul(q_layer.transpose([1, 0, 2]), k_layer.transpose([1, 2, 0]))
        # [b, n, s, s]
        attention_scores = attention_scores.reshape(output_shape)

        if self.scale_mask_softmax:
            self.scale_mask_softmax.scale = attention_scale_coeff
            attention_probs = self.scale_mask_softmax(attention_scores, ltor_mask)
        else:
            if not (ltor_mask == 0).all():
                ltor_mask = ltor_mask.astype(attention_scores.dtype)
                attention_scores = paddle.multiply(attention_scores, 1.0 - ltor_mask)
                attention_scores = attention_scores + (-10000.0) * ltor_mask
            attention_scores = attention_scores.astype("float32") * attention_scale_coeff

            attention_probs = F.softmax(attention_scores, axis=-1)
            attention_probs = attention_probs.astype(self.config.paddle_dtype)

        # [b, n, s, h/n]
        output_shape = [v_layer.shape[1], v_layer.shape[2], q_layer.shape[0], v_layer.shape[3]]
        # [s, b * n, h/n]
        v_layer = v_layer.reshape([v_layer.shape[0], output_shape[0] * output_shape[1], -1])
        # [b * n, s, s]
        attention_probs = attention_probs.reshape([output_shape[0] * output_shape[1], output_shape[2], -1])

        # [b * n, s, h/n]
        context_layer = paddle.bmm(attention_probs, v_layer.transpose([1, 0, 2]))
        context_layer = context_layer.reshape(output_shape)

        # [s, b, n, h/n]
        context_layer = context_layer.transpose([2, 0, 1, 3])

        # [s, b, h]
        new_context_shape = context_layer.shape[:-2] + [self.num_attention_heads * self.attention_head_size]
        context_layer = context_layer.reshape(new_context_shape)

        output = self.dense(context_layer)

        return output, cache_kv, attention_probs


class ChatGLMBlock(nn.Layer):
    """
    The Transformer layer.
    """

    def __init__(self, config: ChatGLMConfig, layer_id: int):
        super(ChatGLMBlock, self).__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)
        self.attention = ChatGLMAttention(config)

        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)
        self.mlp = ChatGLMMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        ltor_mask: Tensor,
        position_ids: Tensor,
        use_cache: bool = False,
        cache: Tensor = None,
    ):
        hidden_states.numpy().dump(f"align_paddle_{hidden_states.dtype}.npy")
        attention_input = self.input_layernorm(hidden_states)
        # Layer norm before transformer layer
        # cache = self.input_layernorm(cache) if cache is not None else None
        # Self attention
        attention_output, cache, _ = self.attention(
            hidden_states=attention_input,
            ltor_mask=ltor_mask,
            position_ids=position_ids,
            cache=cache,
            use_cache=use_cache,
            layer_id=self.layer_id,
        )
        # Residual connection
        alpha = (2 * self.config.num_hidden_layers) ** 0.5
        layernorm_input = alpha * attention_input + attention_output
        # Layernorm after attention
        mlp_input = self.post_attention_layernorm(layernorm_input)
        # MLP
        mlp_output = self.mlp(mlp_input)
        # Second residual connection
        output = mlp_input * alpha + mlp_output
        return output, cache


class ChatGLMMLP(nn.Layer):
    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMMLP, self).__init__()
        self.config = config
        if config.inner_hidden_size is None:
            inner_hidden_size = config.hidden_size * 4
        else:
            inner_hidden_size = config.inner_hidden_size

        if config.tensor_parallel_degree > 1:
            self.dense_h_to_4h = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size, inner_hidden_size, has_bias=True, gather_output=False
            )
            self.dense_4h_to_h = fleet.meta_parallel.RowParallelLinear(
                inner_hidden_size, config.hidden_size, input_is_parallel=True, has_bias=True
            )
        else:
            self.dense_h_to_4h = nn.Linear(config.hidden_size, inner_hidden_size)
            self.dense_4h_to_h = nn.Linear(inner_hidden_size, config.hidden_size)
        # self.dropout = nn.Dropout(config.output_dropout_prob)
        self.activation = self.geglue if self.config.activation == "geglu" else self.gelu

    def geglu(self, x):
        x1, x2 = paddle.chunk(x, chunks=2, axis=-1)
        x = x1 * F.gelu(x2)
        return x

    def gelu(self, x):
        return F.gelu(x, approximate=True)

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        # output = self.dropout(output)
        return output


class ChatGLMStack(nn.Layer):
    """
    GLM Transformer
    """

    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMStack, self).__init__()
        self.config = config
        self.position_encoding_2d = config.position_encoding_2d
        self.hidden_size = config.hidden_size
        self.enable_recompute = config.recompute
        # self.embedding_dropout = nn.Dropout(config.embedding_dropout_prob)

        if self.config.tensor_parallel_degree > 1:
            self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.word_embeddings = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )

        self.layers = nn.LayerList()
        for index in range(config.num_hidden_layers):
            self.layers.append(ChatGLMBlock(config, index))

        self.final_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)

        if self.config.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = paddle.arange(self.pre_seq_len, dtype="int64")
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = nn.Dropout(0.1)

    def get_prompt(self, batch_size, dtype=paddle.float16):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand([batch_size, -1])
        past_key_values = self.prefix_encoder(prefix_tokens).astype(dtype)
        past_key_values = past_key_values.reshape(
            batch_size,
            self.config.pre_seq_len,
            self.config.num_layers * 2,
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads,
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.transpose([2, 1, 0, 3, 4]).split(2)
        # past_key_values = [(v[0], v[1]) for v in past_key_values]
        return past_key_values

    @paddle.jit.not_to_static
    def recompute_training(
        self, layer_module: nn.Layer, hidden_states: Tensor, ltor_mask: Tensor, position_ids: Tensor, cache: Tensor
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(create_custom_forward(layer_module), hidden_states, ltor_mask, position_ids, cache)
        return hidden_states

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        inputs_embeds: Tensor = None,
        past_key_values: Optional[Tensor] = None,
        use_cache: bool = False,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            inputs_embeds = inputs_embeds.transpose([1, 0, 2])

        if past_key_values is None:
            if self.config.pre_seq_len is not None:
                past_key_values = self.get_prompt(batch_size=input_ids.shape[0], dtype=inputs_embeds.dtype)
            else:
                past_key_values = tuple([None] * len(self.layers))

        if self.config.pre_seq_len is not None and attention_mask is not None:
            prefix_attention_mask = paddle.ones([batch_size, 1, input_ids.shape[-1], self.config.pre_seq_len])
            prefix_attention_mask = (prefix_attention_mask < 0.5).astype("bool")
            attention_mask = paddle.concat((prefix_attention_mask, attention_mask), axis=3)

        hidden_states = inputs_embeds

        current_key_values = [] if use_cache else None
        if attention_mask is None:
            attention_mask = paddle.zeros([1, 1]).astype("bool")

        for i, layer in enumerate(self.layers):
            cache_i = past_key_values[i]

            if self.enable_recompute:
                hidden_states, cache = self.recompute_training(
                    layer,
                    hidden_states=hidden_states,
                    ltor_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    cache=cache_i,
                )
            else:
                hidden_states, cache = layer(
                    hidden_states=hidden_states,
                    ltor_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    cache=cache_i,
                )

            if use_cache:
                current_key_values.append(cache.detach())

        output = self.final_layernorm(hidden_states)
        return (output, current_key_values)


class ChatGLMPretrainedModel(PretrainedModel):
    """
    An abstarct class for pretrained ChatGLM models. It provides GLM related
    `model_config_file`, `resource_file_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    base_model_prefix = "chatglm"
    config_class = ChatGLMConfig
    model_config_file = CONFIG_NAME
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = CHATGLM_PRETRAINED_RESOURCE_FILES_MAP

    def init_weights(self, layer):
        """Initialization hook"""
        return None

    def get_masks(self, input_ids):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = paddle.tril(paddle.ones([batch_size, seq_length, seq_length]))
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = (attention_mask < 0.5).astype("bool")
        return attention_mask

    def get_position_ids(self, input_ids, mask_positions, gmask=False):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        if self.position_encoding_2d:
            position_ids = paddle.arange(seq_length, dtype="int64").unsqueeze(0).tile([batch_size, 1])
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [
                paddle.concat(
                    (
                        paddle.zeros([context_length], dtype="int64"),
                        paddle.arange(seq_length - context_length, dtype="int64") + 1,
                    )
                )
                for context_length in context_lengths
            ]
            block_position_ids = paddle.stack(block_position_ids, axis=0)
            position_ids = paddle.stack((position_ids, block_position_ids), axis=1)
        else:
            position_ids = paddle.arange(seq_length, dtype="int64").unsqueeze(0).tile([batch_size, 1])
            if not gmask:
                for i, context_length in enumerate(context_lengths):
                    position_ids[context_length:] = mask_positions[i]

        return position_ids

    @classmethod
    def _get_tensor_parallel_mappings(cls, config):

        import numpy as np

        from paddlenlp.transformers.conversion_utils import (
            naive_merged_qkv_to_tensor_parallel_qkv,
            split_tensor_parallel_weight,
        )

        def fn(x, is_column=True, transpose=False, is_old_qkv=False):
            if transpose:
                x = np.transpose(x, [1, 0])
            if is_old_qkv:
                assert is_column, "QKV vectors should be column parallel linear."
                x = naive_merged_qkv_to_tensor_parallel_qkv(x, config.num_attention_heads)
            return split_tensor_parallel_weight(
                x,
                tensor_parallel_degree=config.tensor_parallel_degree,
                tensor_parallel_rank=config.tensor_parallel_rank,
                is_column=is_column,
            )

        def get_tensor_parallel_split_mappings(num_hidden_layers):
            final_actions = {}
            base_actions = {
                # Column Linear
                "transformer.layers.0.mlp.dense_h_to_4h.bias": partial(fn, is_column=True),
                "transformer.layers.0.mlp.dense_h_to_4h.weight": partial(fn, is_column=True),
                "transformer.layers.0.attention.query_key_value.bias": partial(fn, is_column=True),
                "transformer.layers.0.attention.query_key_value.weight": partial(fn, is_column=True),
                # Row Linear
                "transformer.word_embeddings.weight": partial(fn, is_column=False),
                # 'transformer.layers.0.attention.dense.bias',
                "transformer.layers.0.attention.dense.weight": partial(fn, is_column=False),
                # 'transformer.layers.0.mlp.dense_4h_to_h.bias',
                "transformer.layers.0.mlp.dense_4h_to_h.weight": partial(fn, is_column=False),
            }
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_hidden_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        tp_split_mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        # prefix = ""
        prefix = "chatglm."

        return [StateDictNameMapping(prefix + key, prefix + key, action) for key, action in tp_split_mappings.items()]


@register_base_model
class ChatGLMModel(ChatGLMPretrainedModel):
    r"""
    The GLM Model transformer can behave as an encoder (with only self-attention) as well as a decoder, where
    a layer of cross-attention is added between the self-attention layers, following the architecture
    described in [Attention is all you need](https://arxiv.org/abs/1706.03762).

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    """

    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMModel, self).__init__(config)
        self.config = config
        self.transformer = ChatGLMStack(config)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.transformer.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.transformer.word_embeddings = new_embeddings

    def parallel_matmul(self, lm_output, logit_weights, parallel_output):
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        world_size = hcg.get_model_parallel_world_size()

        if world_size > 1:
            # _c_identity is backwards is reduce
            input_parallel = paddle.distributed.collective._c_identity(lm_output, group=model_parallel_group)

            logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

            if parallel_output:
                return logits

            # _c_concat has not grad backwards
            return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
        else:
            logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
            return logits

    def get_attention_mask(self, input_ids):
        context_length = paddle.where(input_ids == self.config.bos_token_id)[0] + 1
        attention_mask = paddle.tril(paddle.ones([input_ids.shape[0], input_ids.shape[0]]))
        attention_mask[..., : context_length - 1] = 1
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = (attention_mask < 0.5).bool()
        return attention_mask

    def get_position_ids(self, input_ids, mask_position, gmask=False):
        context_length = len(input_ids)
        if self.position_encoding_2d:
            seq_length = paddle.where(input_ids == self.config.bos_token_id)[0]
            position_ids = paddle.arange(context_length, dtype="int64")
            if not gmask:
                position_ids[seq_length:] = mask_position
            block_position_ids = paddle.concat(
                [
                    paddle.zeros(seq_length, dtype="int64"),
                    paddle.arange(context_length - seq_length, dtype="int64") + 1,
                ]
            )
            position_ids = paddle.stack([position_ids, block_position_ids], axis=0)
        else:
            position_ids = paddle.arange(context_length)
            if not gmask:
                position_ids[seq_length:] = mask_position

        position_ids = position_ids.unsqueeze(0)
        return position_ids

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,
        attention_mask: Tensor = None,
        past_key_values=None,
        inputs_embeds: Tensor = None,
        use_cache: bool = None,
        return_dict: bool = None,
    ):
        if attention_mask is None:
            attention_mask = self.get_masks(input_ids)

        if position_ids is None:
            MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
            mask_token = gMASK if sum([gMASK in x for x in input_ids]) > 0 else MASK
            use_gmask = True if sum([gMASK in x for x in input_ids]) > 0 else False

            mask_positions = [seq.tolist().index(mask_token) for seq in input_ids]
            position_ids = self.get_position_ids(input_ids, mask_positions=mask_positions, gmask=use_gmask)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        logits, hidden_layers = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if not return_dict:
            return (logits, hidden_layers)

        return ModelOutput(logits=logits, cache=hidden_layers)


class ChatGLMForConditionalGeneration(ChatGLMPretrainedModel):
    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMForConditionalGeneration, self).__init__(config)

        self.config = config
        self.max_sequence_length = config.max_sequence_length
        self.position_encoding_2d = config.position_encoding_2d
        self.chatglm = ChatGLMModel(config)

        self.lm_head = self.chatglm.get_input_embeddings()

    def prepare_inputs_for_generation(
        self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, cache=None, **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
        mask_token = gMASK if sum([gMASK in x for x in input_ids]) > 0 else MASK
        use_gmask = True if sum([gMASK in x for x in input_ids]) > 0 else MASK
        seqs = input_ids.tolist()
        mask_positions = [seq.index(mask_token) for seq in seqs]

        if cache is not None or past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None and attention_mask.dtype == paddle.bool:
                attention_mask = attention_mask[:, :, -1:]
            else:
                attention_mask = None
            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            else:
                context_lengths = [seq.index(self.config.bos_token_id) for seq in seqs]
                if self.position_encoding_2d:
                    position_ids = paddle.to_tensor(
                        [
                            [mask_position, seq_length - context_length]
                            for mask_position, context_length in zip(mask_positions, context_lengths)
                        ],
                        dtype="int64",
                    ).unsqueeze(-1)
                else:
                    position_ids = paddle.to_tensor(
                        [mask_position for mask_position in mask_positions], dtype="int64"
                    ).unsqueeze(-1)

            if cache is None:
                cache = past_key_values
            return {
                "input_ids": last_token,
                "past_key_values": cache,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
        else:
            if attention_mask is not None and attention_mask.dtype != paddle.bool:
                logger.warning(f"The dtype of attention mask ({attention_mask.dtype}) is not bool")
                attention_mask = None
            if attention_mask is None:
                attention_mask = self.get_masks(input_ids)
            if position_ids is None:
                position_ids = self.get_position_ids(input_ids, mask_positions=mask_positions, gmask=use_gmask)

            return {
                "input_ids": input_ids,
                "past_key_values": cache,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs["past_key_values"] if hasattr(outputs, "past_key_values") else None

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if attention_mask is not None and attention_mask.dtype == paddle.bool:
                attention_mask = paddle.concat(
                    [attention_mask, paddle.ones([*attention_mask.shape[:3], 1], dtype=paddle.bool)], axis=3
                )
                new_attention_mask = attention_mask[:, :, -1:].clone()
                new_attention_mask[..., -1] = False
                model_kwargs["attention_mask"] = paddle.concat([attention_mask, new_attention_mask], axis=2)

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id[:, 1, :] += 1
            model_kwargs["position_ids"] = paddle.concat([position_ids, new_position_id], axis=-1)

        return model_kwargs

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_dict=True,
    ):
        transformer_outputs = self.chatglm(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs.logits if return_dict else transformer_outputs[0]

        lm_logits = F.linear(hidden_states, self.lm_head.weight.T).transpose([1, 0, 2])
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :]
            shift_logits = shift_logits.reshape([-1, shift_logits.shape[-1]])
            shift_labels = labels[..., 1:].reshape([-1])
            loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        if not return_dict:
            if loss is not None:
                return (loss, lm_logits, transformer_outputs[1:])
            else:
                return (lm_logits, transformer_outputs.cache)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.cache,
        )

    @staticmethod
    def _reorder_cache(cache, beam_idx):
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx),
                layer_past[1].index_select(1, beam_idx),
            )
            for layer_past in cache
        )

    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response
