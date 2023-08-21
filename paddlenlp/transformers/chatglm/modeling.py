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
from paddle import Tensor
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute

from ...utils.env import CONFIG_NAME
from ...utils.log import logger
from .. import PretrainedModel, register_base_model
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPast,
)
from .configuration import CHATGLM_PRETRAINED_RESOURCE_FILES_MAP, ChatGLMConfig

__all__ = [
    "ChatGLMModel",
    "ChatGLMPretrainedModel",
    "ChatGLMForCausalLM",
]


def parallel_matmul(lm_output, logit_weights, parallel_output):
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
    def __init__(self, hidden_size, base=10000.0, position_encoding_2d=True):
        super().__init__()
        self.default_dtype = paddle.get_default_dtype()
        inv_freq = 1.0 / (base ** (paddle.arange(0, hidden_size, 2).astype("float32") / hidden_size))
        inv_freq = inv_freq.astype(self.default_dtype)
        self.position_encoding_2d = position_encoding_2d
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = -1
        self.cos_cached = None
        self.sin_cached = None

    def get_rotary_embeds(self, cos, sin, position_ids):
        # [s, b, 1, h/n]
        cos = cos.squeeze(1)[position_ids].unsqueeze(2)
        sin = sin.squeeze(1)[position_ids].unsqueeze(2)
        return paddle.stack([cos, sin], axis=0)

    def forward(self, position_ids):

        seq_len = position_ids.max() + 1
        # seq_len = position_ids.shape[-1]

        if self.max_seq_len_cached < 0 or seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            # x.shape = [b, s, n, h/n/2]
            # TODO(duanyanhui): npu arange kernel don't support fp16, and
            # it can't be fallbacked to cpu. It will be fixed in future.
            if paddle.get_device().split(":")[0] == "npu":
                t = paddle.arange(start=0, end=seq_len, dtype="float32")
                t = t.cast(self.inv_freq.dtype)
            else:
                t = paddle.arange(start=0, end=seq_len, dtype=self.inv_freq.dtype)
            # [s, h/n/2]
            if not paddle.in_dynamic_mode():
                inv_freq = paddle.cast(self.inv_freq, "float32")
                t = paddle.cast(t, "float32")
                freqs = paddle.einsum("i,j->ij", t, inv_freq)
            else:
                freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
            freqs = freqs.cast(self.default_dtype)
            # [s, h/n]
            emb = paddle.concat([freqs, freqs], axis=-1)
            if self.default_dtype == paddle.bfloat16:
                emb = emb.cast("float32")
            # [s, 1, h/n]
            cos_cached = emb.cos().unsqueeze(1)
            sin_cached = emb.sin().unsqueeze(1)

            if self.default_dtype == paddle.bfloat16:
                cos_cached = cos_cached.astype(self.default_dtype)
                sin_cached = sin_cached.astype(self.default_dtype)

            if hasattr(paddle.framework, "_no_check_dy2st_diff"):
                # TODO(daisiming): _no_check_dy2st_diff is used to turn off the checking of behavior
                # inconsistency between dynamic graph and static graph. _no_check_dy2st_diff should be
                # removed after static graphs support inplace and stride.
                with paddle.framework._no_check_dy2st_diff():
                    self.cos_cached, self.sin_cached = cos_cached, sin_cached
            else:
                self.cos_cached, self.sin_cached = cos_cached, sin_cached

        cos, sin = self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]
        if self.position_encoding_2d:
            block_position_ids = position_ids[:, 1, :].transpose([1, 0])
            position_ids = position_ids[:, 0, :].transpose([1, 0])
            block_rotary_embeds = self.get_rotary_embeds(cos, sin, block_position_ids)
            position_rotary_embeds = self.get_rotary_embeds(cos, sin, position_ids)
            rotary_embeds = paddle.stack([position_rotary_embeds, block_rotary_embeds], axis=0)
        else:
            position_ids = position_ids.transpose([1, 0])
            rotary_embeds = self.get_rotary_embeds(cos, sin, position_ids)

        return rotary_embeds


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
        self.scale_mask_softmax = False
        self.default_dtype = paddle.get_default_dtype()

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

    def _apply_rotary_position_embed_index(self, q, k, cos, sin):
        # q.shape = [s, b, n, h/n/2], cos.shape = [s, 1, h/n], position_ids.shape = [s, b]
        # [s, b, n, h/n]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    def _core_attention(self, q_layer: Tensor, k_layer: Tensor, position_ids: Tensor, rotary_embeds: Tensor):
        # Set store_true, position_encoding_2d=False by default.
        if self.config.position_encoding_2d:
            # [s, b, n, h/n/2]
            q1, q2 = paddle.chunk(q_layer, 2, axis=-1)
            k1, k2 = paddle.chunk(k_layer, 2, axis=-1)

            pcos, psin = rotary_embeds[0][0], rotary_embeds[0][1]
            bcos, bsin = rotary_embeds[1][0], rotary_embeds[1][1]

            # [s, b, n, h/n]
            q1, k1 = self._apply_rotary_position_embed_index(q1, k1, pcos, psin)
            q2, k2 = self._apply_rotary_position_embed_index(q2, k2, bcos, bsin)
            q_layer = paddle.concat([q1, q2], axis=-1)
            k_layer = paddle.concat([k1, k2], axis=-1)
        else:
            cos, sin = rotary_embeds[0], rotary_embeds[1]
            # [s, b, n, h/n]
            q_layer, k_layer = self._apply_rotary_position_embed_index(q_layer, k_layer, cos, sin)
        return q_layer, k_layer

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        use_cache: bool = False,
        cache: Tensor = None,
        layer_id=0,
        rotary_embeds=None,
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
        q_layer, k_layer = self._core_attention(q_layer, k_layer, position_ids, rotary_embeds)

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
            q_layer = q_layer.astype(self.default_dtype)

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
            attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        else:
            attention_scores = attention_scores + attention_mask
            attention_scores = attention_scores.astype("float32")
            attention_scores = attention_scores * attention_scale_coeff
            attention_probs = F.softmax(attention_scores, axis=-1)
            attention_probs = attention_probs.astype(self.default_dtype)
            v_layer = v_layer.astype(self.default_dtype)

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
        self.default_dtype = paddle.get_default_dtype()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)
        self.attention = ChatGLMAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)
        self.mlp = ChatGLMMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        use_cache: bool = False,
        cache: Tensor = None,
        rotary_embeds: Tensor = None,
    ):
        # Layer norm before transformer layer
        attention_input = self.input_layernorm(hidden_states)
        # Self attention
        attention_output, cache, _ = self.attention(
            hidden_states=attention_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            use_cache=use_cache,
            layer_id=self.layer_id,
            rotary_embeds=rotary_embeds,
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
        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.num_attention_heads = config.num_attention_heads
        self.rotary_embeddings = RotaryEmbeddings(
            self.hidden_size // (self.num_attention_heads * 2)
            if self.position_encoding_2d
            else self.hidden_size // self.num_attention_heads,
            base=10000.0,
        )
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
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        use_cache: bool,
        cache: Tensor,
        rotary_embeds: Tensor,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            use_cache,
            cache,
            rotary_embeds,
            use_reentrant=False,
        )
        return hidden_states

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        inputs_embeds: Tensor = None,
        cache: Optional[Tensor] = None,
        use_cache: bool = False,
    ):

        if input_ids is not None and inputs_embeds is not None:
            input_ids = None
            logger.warning("Specify both input_ids and inputs_embeds at the same time, will use inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        inputs_embeds = inputs_embeds.transpose([1, 0, 2])

        rotary_embeds = self.rotary_embeddings(position_ids)

        if cache is None:
            if self.config.pre_seq_len is not None:
                cache = self.get_prompt(batch_size=input_ids.shape[0], dtype=inputs_embeds.dtype)
            else:
                cache = tuple([None] * len(self.layers))

        # this branch is deprecated
        if self.config.pre_seq_len is not None and attention_mask is not None:
            prefix_attention_mask = paddle.ones([batch_size, 1, input_ids.shape[-1], self.config.pre_seq_len])
            prefix_attention_mask = (prefix_attention_mask < 0.5).astype("int64")
            attention_mask = paddle.concat((prefix_attention_mask, attention_mask), axis=3)

        zero = paddle.zeros(attention_mask.shape, dtype=inputs_embeds.dtype)
        neg_inf = paddle.full_like(attention_mask, paddle.finfo(inputs_embeds.dtype).min, dtype=inputs_embeds.dtype)
        attention_mask = paddle.where(attention_mask, zero, neg_inf)

        hidden_states = inputs_embeds

        current_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            cache_i = cache[i]

            if self.enable_recompute and not hidden_states.stop_gradient:
                hidden_states, new_cache = self.recompute_training(
                    layer,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    cache=cache_i,
                    rotary_embeds=rotary_embeds,
                )
            else:
                hidden_states, new_cache = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    cache=cache_i,
                    rotary_embeds=rotary_embeds,
                )

            if use_cache:
                current_caches.append(new_cache)

        output = self.final_layernorm(hidden_states)
        return (output, current_caches)


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
    _keys_to_ignore_on_load_missing = [r"transformer.rotary_embeddings.inv_freq", r"lm_head.decoder_weight"]
    _keys_to_ignore_on_load_unexpected = [r"transformer.rotary_emb.inv_freq"]

    def init_weights(self, layer):
        """Initialization hook"""
        return None

    def get_position_ids(self, input_ids, mask_positions, use_gmasks=None):
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size

        context_lengths = []
        for seq in input_ids:
            context_lengths.append(paddle.where(seq == self.config.bos_token_id)[0][0])

        if self.config.position_encoding_2d:
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
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[context_length:] = mask_positions[i]

        return position_ids

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None, None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, 2, None], dtype="int64"),
        }

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
                "transformer.layers.0.mlp.dense_h_to_4h.bias": partial(fn, is_column=True),
                "transformer.layers.0.mlp.dense_h_to_4h.weight": partial(fn, is_column=True),
                "transformer.layers.0.attention.query_key_value.bias": partial(fn, is_column=True),
                "transformer.layers.0.attention.query_key_value.weight": partial(fn, is_column=True),
                # Row Linear
                "transformer.word_embeddings.weight": partial(fn, is_column=False),
                "transformer.layers.0.attention.dense.weight": partial(fn, is_column=False),
                "transformer.layers.0.mlp.dense_4h_to_h.weight": partial(fn, is_column=False),
            }
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_hidden_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings


@register_base_model
class ChatGLMModel(ChatGLMPretrainedModel):
    r"""
    The GLM Model transformer can behave as an encoder (with only self-attention) as well as a decoder, where
    a layer of cross-attention is added between the self-attention layers, following the architecture
    described in [Attention is all you need](https://arxiv.org/abs/1706.03762).

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    """
    _keys_to_ignore_on_load_unexpected = [r"transformer.layers.*.attention.rotary_emb.inv_freq", r"lm_head.weight"]

    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMModel, self).__init__(config)
        self.config = config
        self.transformer = ChatGLMStack(config)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.transformer.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.transformer.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,
        attention_mask: Tensor = None,
        cache=None,
        inputs_embeds: Tensor = None,
        use_cache: bool = None,
        return_dict: bool = None,
    ):
        if input_ids is None:
            assert position_ids is not None, "`position_ids` must be explicitly specified when input_ids is None."
            assert attention_mask is not None, "`attention_mask` must be explicitly specified when input_ids is None."

        if attention_mask is None or len(attention_mask.shape) != 4:
            raise ValueError(f"attention mask should'nt be None or has size other than 4Dim. Found {attention_mask}")

        attention_mask = attention_mask.astype("bool")

        if position_ids is None:
            MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id

            use_gmasks = []
            mask_positions = []
            for seq in input_ids:
                mask_token = gMASK if gMASK in seq else MASK
                use_gmask = mask_token == gMASK
                use_gmasks.append(use_gmask)
                mask_positions.append(paddle.where(seq == mask_token)[0][0])
            position_ids = self.get_position_ids(input_ids, mask_positions=mask_positions, use_gmasks=use_gmasks)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        logits, new_caches = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache=cache,
            use_cache=use_cache,
        )

        if not return_dict:
            return (logits, new_caches)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=logits, past_key_values=new_caches)


class ChatGLMHead(nn.Layer):
    def __init__(self, config, embedding_weights=None):
        super(ChatGLMHead, self).__init__()
        self.decoder_weight = (
            self.create_parameter(shape=[config.vocab_size, config.hidden_size], dtype=paddle.get_default_dtype())
            if embedding_weights is None
            else embedding_weights
        )
        self.config = config

    def forward(self, hidden_states):
        if self.config.tensor_parallel_degree > 1:
            logits = parallel_matmul(hidden_states, self.decoder_weight, self.config.tensor_parallel_output)
        else:
            logits = F.linear(hidden_states, self.decoder_weight.T)
        return logits


class ChatGLMForCausalLM(ChatGLMPretrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder_weight"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMForCausalLM, self).__init__(config)

        self.config = config
        self.max_sequence_length = config.max_sequence_length
        self.position_encoding_2d = config.position_encoding_2d
        self.chatglm = ChatGLMModel(config)

        self.lm_head = ChatGLMHead(config, self.chatglm.transformer.word_embeddings.weight)
        # from paddlenlp.transformers import ChatGLMTokenizer
        # self.tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b")

    def prepare_inputs_for_generation(
        self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, cache=None, **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
        use_gmasks = []
        mask_positions = []
        for seq in input_ids:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            use_gmasks.append(use_gmask)
            mask_positions.append(paddle.where(seq == mask_token)[0][0])

        if cache is not None or past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)

            attention_mask = attention_mask[:, :, -1:]

            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            else:
                if self.position_encoding_2d:
                    context_lengths = []
                    for seq in input_ids:
                        context_lengths.append(paddle.where(seq == self.config.bos_token_id)[0][0])

                    context_lengths = paddle.to_tensor(context_lengths, dtype="int64")
                    block_position_ids = seq_length - context_lengths
                    position_ids = paddle.concat(
                        [paddle.to_tensor(mask_positions, dtype="int64"), block_position_ids], axis=1
                    ).unsqueeze(-1)
                else:
                    position_ids = paddle.to_tensor(mask_positions, dtype="int64").unsqueeze(-1)

            if cache is None:
                cache = past_key_values
            return {
                "input_ids": last_token,
                "cache": cache[-1],
                "position_ids": position_ids,
                "use_cache": True,
                "attention_mask": attention_mask,
            }
        else:
            if position_ids is None:
                position_ids = self.get_position_ids(input_ids, mask_positions=mask_positions, use_gmasks=use_gmasks)

            return {
                "input_ids": input_ids,
                "cache": cache,
                "position_ids": position_ids,
                "use_cache": True,
                "attention_mask": attention_mask,
            }

    def update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update cache
        model_kwargs["cache"] = outputs[1] if isinstance(outputs, tuple) else outputs["past_key_values"]

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if attention_mask is not None:
                attention_mask = paddle.concat(
                    [attention_mask, paddle.zeros([*attention_mask.shape[:3], 1], attention_mask.dtype)], axis=3
                )
                new_attention_mask = attention_mask[:, :, -1:].clone()
                new_attention_mask[..., -1] = 1
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
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        cache=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_dict=False,
    ):
        transformer_outputs = self.chatglm(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs.last_hidden_state if return_dict else transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        lm_logits = lm_logits.transpose([1, 0, 2]).astype("float32")
        loss = None
        if labels is not None:
            if self.config.tensor_parallel_degree > 1 and self.config.tensor_parallel_output:
                self.parallel_loss_func = fleet.meta_parallel.ParallelCrossEntropy()
                filtered_logits = lm_logits[labels != -100]
                filtered_labels = labels[labels != -100]
                loss = self.parallel_loss_func(filtered_logits, filtered_labels).mean()
            else:
                loss = nn.functional.cross_entropy(lm_logits, labels, ignore_index=-100)
            loss = loss.astype(lm_logits.dtype)

        if not return_dict:
            if loss is not None:
                return (loss, lm_logits, transformer_outputs[1:])
            else:
                return (lm_logits, transformer_outputs[1:])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
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

    @staticmethod
    def process_response(response):
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
