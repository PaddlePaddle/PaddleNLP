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
from functools import partial
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from configuration import (
    GLM130B_PRETRAINED_INIT_CONFIGURATION,
    GLM130B_PRETRAINED_RESOURCE_FILES_MAP,
    GLM130BConfig,
)
from paddle import Tensor
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.transformers.model_outputs import ModelOutput
from paddlenlp.utils.converter import StateDictNameMapping
from paddlenlp.utils.env import CONFIG_NAME
from paddlenlp.utils.initializer import normal_, ones_, zeros_

__all__ = [
    "GLM130BModel",
    "GLM130BPretrainedModel",
]


class RotaryEmbeddings(nn.Layer):
    def __init__(self, hidden_size, base=10000, dtype=paddle.float16):
        super().__init__()
        inv_freq = 1.0 / (base ** (paddle.arange(0, hidden_size, 2).astype("float32") / hidden_size))
        inv_freq = inv_freq.astype(dtype)
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
            self.max_seq_len_cached = seq_len
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

            self.cos_cached, self.sin_cached = cos_cached, sin_cached

        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class GLM130BAttention(nn.Layer):
    """
    Self-attention layer performs multiple attention to jointly attending to
    information from different representation subspaces.
    """

    def __init__(self, config: GLM130BConfig):
        super(GLM130BAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
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

        self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
        self.output_dropout = nn.Dropout(config.output_dropout_prob)

    def _transpose_for_scores(self, inputs: Tensor):
        """
        Transpose a 3D tensor [b, s, n/p*h/n] into a 4D tensor [b, n/p, s, h/n],
        where b means batch_size, s means sequence_length, n means num_attention_heads,
        h means hidden_size and p means number of partitions.
        """
        new_shape = [*inputs.shape[:2], self.num_attention_heads, self.attention_head_size]
        outputs = inputs.reshape(new_shape)
        outputs = paddle.transpose(outputs, [1, 0, 2, 3])
        return outputs

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

    def _core_attention(
        self, q_layer: Tensor, k_layer: Tensor, position_ids: Tensor, rotary_embeddings: RotaryEmbeddings = None
    ):
        # Set store_true, position_encoding_2d=False by default.
        if self.config.position_encoding_2d:
            # [s, b, n, h/n/2]
            q1, q2 = paddle.chunk(q_layer, 2, axis=-1)
            k1, k2 = paddle.chunk(k_layer, 2, axis=-1)
            # [s, 1, h/n]
            cos, sin = rotary_embeddings(q1, seq_len=position_ids.max() + 1)
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
            cos, sin = rotary_embeddings(q_layer, seq_len=position_ids.max() + 1)
            # [s, b, n, h/n]
            q_layer, k_layer = self._apply_rotary_position_embed_index(q_layer, k_layer, cos, sin, position_ids)
        return q_layer, k_layer

    def forward(
        self,
        hidden_states: Tensor,
        ltor_mask: Tensor,
        position_ids: Tensor,
        rotary_embeddings: Tensor,
        use_cache: bool = False,
        cache: Tensor = None,
        label_id=0,
    ):
        hidden_states = hidden_states.transpose([1, 0, 2])
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
        q_layer, k_layer = self._core_attention(q_layer, k_layer, position_ids, rotary_embeddings)
        seq_length, batch_size, num_heads, hidden_size = k_layer.shape
        cache_kv = None
        if use_cache:
            cache_kv = (
                paddle.stack([k_layer, v_layer], axis=0)
                .transpose([2, 1, 0, 3, 4])
                .reshape([batch_size, seq_length, num_heads * hidden_size * 2])
            )
        if use_cache and cache is not None:
            cache = cache.expand([batch_size, -1, -1]).reshape([batch_size, cache.shape[1], 2, num_heads, hidden_size])
            # [2, c, b, n, h/n]
            cache = cache.transpose([2, 1, 0, 3, 4])
            cache_k, cache_v = cache[0], cache[1]
            # [s + c, b, n, h/n]
            k_layer = paddle.concat([cache_k, k_layer], axis=0)
            v_layer = paddle.concat([cache_v, v_layer], axis=0)

        attention_scale_coeff = float(label_id) + 1.0
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

        ltor_mask = ltor_mask.astype(attention_scores.dtype)
        attention_scores = paddle.multiply(attention_scores, 1.0 - ltor_mask)
        attention_scores = attention_scores + (-10000.0) * ltor_mask
        attention_scores = attention_scores.astype("float32") * attention_scale_coeff

        attention_probs = F.softmax(attention_scores, axis=-1)
        attention_probs = self.attention_dropout(attention_probs).astype(self.config.paddle_dtype)

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
        output = self.output_dropout(output)
        return output, cache_kv


class GLM130BBlock(nn.Layer):
    """
    The Transformer layer.
    """

    def __init__(self, config: GLM130BConfig, layer_id: int):
        super(GLM130BBlock, self).__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)
        self.attention = GLM130BAttention(config)

        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)
        self.mlp = GLM130BMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        ltor_mask: Tensor,
        position_ids: Tensor,
        rotary_embeddings: Tensor,
        use_cache: bool = False,
        cache: Tensor = None,
    ):
        layernorm_output = self.input_layernorm(hidden_states)
        # Layer norm before transformer layer
        cache = self.input_layernorm(cache) if cache is not None else None
        # Self attention
        attention_output, cache = self.attention(
            layernorm_output, ltor_mask, position_ids, rotary_embeddings, cache, self.layer_id
        )
        # Residual connection
        alpha = (2 * self.config.num_hidden_layers) ** 0.5
        layernorm_input = alpha * layernorm_output + attention_output
        # Layernorm after attention
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection
        output = layernorm_output * alpha + mlp_output
        return output, cache


class GLM130BMLP(nn.Layer):
    def __init__(self, config: GLM130BConfig):
        super(GLM130BMLP, self).__init__()
        inner_hidden_size = config.hidden_size * 4 // 3 * 2
        if config.tensor_parallel_degree > 1:
            self.dense_h_to_4h = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size, inner_hidden_size * 2, has_bias=True, gather_output=False
            )
            self.dense_4h_to_h = fleet.meta_parallel.RowParallelLinear(
                inner_hidden_size, config.hidden_size, input_is_parallel=True, has_bias=True
            )
        else:
            self.dense_h_to_4h = nn.Linear(config.hidden_size, inner_hidden_size * 2)
            self.dense_4h_to_h = nn.Linear(inner_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.output_dropout_prob)
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
        output = self.dropout(output)
        return output


class GLM130BStack(nn.Layer):
    """
    GLM Transformer
    """

    def __init__(self, config: GLM130BConfig):
        super(GLM130BStack, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.enable_recompute = config.recompute
        self.embedding_dropout = nn.Dropout(config.embedding_dropout_prob)

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

        attention_head_size = config.hidden_size // config.num_attention_heads
        if config.position_encoding_2d:
            attention_head_size = attention_head_size // 2
        self.rotary_embeddings = RotaryEmbeddings(attention_head_size, base=10000, dtype=config.paddle_dtype)
        self.layers = nn.LayerList()
        for index in range(config.num_hidden_layers):
            self.layers.append(GLM130BBlock(config, index))

        self.final_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)

    @paddle.jit.not_to_static
    def recompute_training(
        self, layer_module: nn.Layer, hidden_states: Tensor, ltor_mask: Tensor, position_ids: Tensor, cache: Tensor
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module), hidden_states, ltor_mask, position_ids, self.rotary_embeddings, cache
        )
        return hidden_states

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        memory_states: Optional[Tensor] = None,
    ):

        batch_size, query_length = input_ids.shape
        word_embeddings = self.word_embeddings(input_ids)
        # memory_length = memory_states[0].shape[1] if memory_states is not None else 0

        if attention_mask is None:
            attention_mask = paddle.ones([1, 1])

        hidden_states = self.embedding_dropout(word_embeddings)

        memory_layers = [hidden_states.detach()]
        for i, layer in enumerate(self.layers):
            mem_i = memory_states[i] if memory_states else None

            if self.enable_recompute:
                hidden_states = self.recompute_training(
                    layer, hidden_states, attention_mask, position_ids, cache=mem_i
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, position_ids, self.rotary_embeddings, cache=mem_i)

            memory_layers.append(hidden_states.detach())

        output = self.final_layernorm(hidden_states)
        memory_layers = self.update_memories(memory_layers, memory_states)
        return (output, memory_layers)

    def update_memories(self, hiddens, cache):
        memory_length = cache[0].shape[1] if cache else 0
        query_length = hiddens[0].shape[1]
        new_memory_length = memory_length + query_length

        new_memories = []
        for i in range(len(hiddens)):
            if new_memory_length <= query_length or cache is None:
                new_memories.append(hiddens[i][-new_memory_length:])
            else:
                new_memories.append(paddle.concat([cache[i][:, -memory_length:], hiddens[i]], axis=1))
        return new_memories


class GLM130BPretrainedModel(PretrainedModel):
    """
    An abstarct class for pretrained GLM models. It provides GLM related
    `model_config_file`, `resource_file_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    base_model_prefix = "glm-130b"
    config_class = GLM130BConfig
    model_config_file = CONFIG_NAME
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_init_configuration = GLM130B_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = GLM130B_PRETRAINED_RESOURCE_FILES_MAP

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, nn.Linear):
            normal_(layer.weight, mean=0.0, std=self.config.initializer_range)
            if layer.bias is not None:
                zeros_(layer.bias)
        elif isinstance(layer, nn.Embedding):
            normal_(layer.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(layer, nn.LayerNorm):
            ones_(layer.weight)
            zeros_(layer.bias)

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

        prefix = ""
        # prefix = "glm."

        return [StateDictNameMapping(prefix + key, prefix + key, action) for key, action in tp_split_mappings.items()]


@register_base_model
class GLM130BModel(GLM130BPretrainedModel):
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

    def __init__(self, config: GLM130BConfig):
        super(GLM130BModel, self).__init__(config)
        self.config = config
        self.output_predict = config.output_predict
        self.transformer = GLM130BStack(config)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

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
        cache: Tensor = None,
        use_cache: bool = None,
        return_dict: bool = None,
    ):
        batch_size, seq_len = input_ids.shape[:2]
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
            seq = input_ids[0].tolist()

            if attention_mask is None:
                attention_mask = self.get_masks(seq)

            if position_ids is None:
                MASK, gMASK = 150000, 150001
                mask_token = MASK if MASK in input_ids else gMASK
                use_gmask = False if MASK in input_ids else gMASK

                mask_position = paddle.where(seq == mask_token)[0]
                position_ids = self.get_position_ids(seq=seq, mask_position=mask_position, gmask=use_gmask)

        logits, hidden_layers = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            memory_states=cache,
        )
        if self.output_predict:
            if self.config.tensor_parallel_degree > 1:
                # FIXME: @ZHUI fix for jit_to_static
                logits = self.parallel_matmul(
                    logits, self.transformer.word_embeddings.weight, self.config.tensor_parallel_output
                )
            else:
                logits = F.linear(logits, self.transformer.word_embeddings.weight.T)

            # logits = logits.transpose([1, 0, 2])

        if not return_dict:
            return (logits, hidden_layers)

        return ModelOutput(logits=logits, cache=hidden_layers)
