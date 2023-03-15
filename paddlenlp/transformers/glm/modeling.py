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
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed.fleet.utils import recompute

from ...utils.converter import StateDictNameMapping
from ...utils.env import CONFIG_NAME
from ...utils.initializer import normal_, ones_, zeros_
from .. import PretrainedModel, register_base_model
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MultipleChoiceModelOutput,
)
from .configuration import (
    GLM_PRETRAINED_INIT_CONFIGURATION,
    GLM_PRETRAINED_RESOURCE_FILES_MAP,
    GLMConfig,
)

__all__ = [
    "GLMModel",
    "GLMPretrainedModel",
    "GLMForMultipleChoice",
    "GLMForConditionalGeneration",
]


class GLMAttention(nn.Layer):
    """
    Self-attention layer performs multiple attention to jointly attending to
    information from different representation subspaces.
    """

    def __init__(self, config: GLMConfig):
        super(GLMAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_scale = config.attention_scale

        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.output_dropout_prob)

    def _transpose_for_scores(self, inputs: Tensor):
        """
        Transpose a 3D tensor [b, s, n/p*h/n] into a 4D tensor [b, n/p, s, h/n],
        where b means batch_size, s means sequence_length, n means num_attention_heads,
        h means hidden_size and p means number of partitions.
        """
        new_shape = [*inputs.shape[:-1], self.num_attention_heads, self.attention_head_size]
        outputs = inputs.reshape(new_shape)
        outputs = paddle.transpose(outputs, [0, 2, 1, 3])
        return outputs

    def forward(self, hidden_states: Tensor, ltor_mask: Tensor, cache: Tensor = None):
        query_length = hidden_states.shape[1]
        if cache is None:
            mixed_layer = self.query_key_value(hidden_states)
            mixed_q_layer, mixed_k_layer, mixed_v_layer = paddle.split(mixed_layer, 3, axis=-1)
        else:
            concat_hidden_states = paddle.concat([cache, hidden_states], axis=1)
            mixed_layer = self.query_key_value(concat_hidden_states)
            mixed_q_layer, mixed_k_layer, mixed_v_layer = paddle.split(mixed_layer, 3, axis=-1)
            mixed_q_layer = mixed_q_layer[:, -query_length:]

        q_layer = self._transpose_for_scores(mixed_q_layer)
        k_layer = self._transpose_for_scores(mixed_k_layer)
        v_layer = self._transpose_for_scores(mixed_v_layer)

        if self.attention_scale > 1.0:
            attention_scores = paddle.matmul(
                q_layer / math.sqrt(self.attention_scale),
                k_layer.transpose([0, 1, 3, 2]) / math.sqrt(self.attention_head_size * self.attention_scale),
            )
        else:
            attention_scores = paddle.matmul(
                q_layer, k_layer.transpose([0, 1, 3, 2]) / math.sqrt(self.attention_head_size)
            )

        ltor_mask = ltor_mask.astype(attention_scores.dtype)
        attention_scores = paddle.multiply(attention_scores, ltor_mask)
        if self.attention_scale > 1.0:
            max_attention_scores = attention_scores.max(axis=-1, keepdim=True)[0]
            attention_scores -= max_attention_scores
            attention_scores *= self.attention_scale

        attention_scores = attention_scores + (-65504.0) * (1.0 - ltor_mask)
        attention_probs = F.softmax(attention_scores, axis=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, v_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_shape = context_layer.shape[:-2] + [self.hidden_size]
        context_layer = context_layer.reshape(new_context_shape)

        output = self.dense(context_layer)
        output = self.output_dropout(output)

        output = (output, cache)

        return output


class GLMBlock(nn.Layer):
    """
    The Transformer layer.
    """

    def __init__(self, config: GLMConfig):
        super(GLMBlock, self).__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)
        self.attention = GLMAttention(config)

        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states: Tensor, ltor_mask: Tensor, cache: Tensor = None):
        layernorm_output = self.input_layernorm(hidden_states)
        # Layer norm before transformer layer
        if cache is not None:
            cache = self.input_layernorm(cache)
        else:
            cache = None
        # Self attention
        attention_output, cache = self.attention(layernorm_output, ltor_mask, cache)
        # Residual connection
        layernorm_input = hidden_states + attention_output
        # Layernorm after attention
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection
        output = layernorm_input + mlp_output

        return (output, cache)


class GPT2MLP(nn.Layer):
    """
    MLP takes the input with an h hidden state, project it to 4*h hidden
    dimension, perform gelu transformation, and project the state back
    into h hidden dimension. At the end, dropout is also applied.
    """

    def __init__(self, config: GLMConfig):
        super(GPT2MLP, self).__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.dense_4h_to_h = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.output_dropout_prob)

    def forward(self, hidden_states):
        # [batch_size, sequence_length, 4h / number of partitions]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = F.gelu(intermediate_parallel, approximate=True)

        # [batch_size, sequence_length, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class GLMStack(nn.Layer):
    """
    GLM Transformer
    """

    def __init__(self, config: GLMConfig):
        super(GLMStack, self).__init__()
        self.hidden_size = config.hidden_size
        self.enable_recompute = config.checkpoint_activations
        self.checkpoint_num_layers = config.checkpoint_num_layers

        self.embedding_dropout = nn.Dropout(config.embedding_dropout_prob)
        self.block_position_encoding = config.block_position_encoding

        if self.block_position_encoding:
            self.position_embeddings = nn.Embedding(
                config.max_sequence_length + 1,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0, std=config.initializer_range)),
            )
            self.block_position_embeddings = nn.Embedding(
                config.max_sequence_length + 1,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0, std=config.initializer_range)),
            )
        else:
            self.position_embeddings = nn.Embedding(
                config.max_sequence_length,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0, std=config.initializer_range)),
            )

        self.layers = nn.LayerList()
        for _ in range(config.num_layers):
            self.layers.append(GLMBlock(config))

        self.final_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)

    @paddle.jit.not_to_static
    def recompute_training(self, layer_module: nn.Layer, hidden_states: Tensor, ltor_mask: Tensor, cache: Tensor):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(create_custom_forward(layer_module), hidden_states, ltor_mask, cache)
        return hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        cache: Optional[Tensor] = None,
        return_dict: bool = False,
    ):
        batch_size, query_length = hidden_states.shape[:2]
        memory_length = cache[0].shape[1] if cache is not None else 0

        is_scalar = (paddle.numel(attention_mask) == 1)[0]
        is_sep = is_scalar or paddle.numel(attention_mask) == batch_size
        if is_sep:
            scalar_sep = attention_mask[0] if is_scalar else attention_mask

            # attention mask is the beginning postion of B region in [0, query_len)
            def build_mask_matrix(seq_length, sep, memory_length=0):
                mask = paddle.ones([1, seq_length, seq_length])
                mask = paddle.tril(mask)
                if is_scalar:
                    mask[0, :, : int(sep)] = 1
                else:
                    mask = mask.expand([batch_size, -1, -1])
                    ids = paddle.arange(seq_length, dtype=sep.dtype).unsqueeze(0)
                    m = (ids < sep.reshape([-1, 1])).astype("float32")
                    m = m.unsqueeze(1).expand_as(mask).astype("bool")
                    y = paddle.full(mask.shape, 1, mask.dtype)
                    mask = paddle.where(m, y, mask)
                if memory_length > 0:
                    mask = mask.expand([batch_size, -1, -1])
                    mask = paddle.concat([paddle.ones([batch_size, seq_length, memory_length]), mask], axis=2)
                mask = mask.unsqueeze(1)
                return mask

            attention_mask = build_mask_matrix(query_length, scalar_sep, memory_length=memory_length)
        else:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask[:, :, :, -query_length - memory_length :]

        if self.block_position_encoding:
            position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = self.position_embeddings(position_ids)

        hidden_states = hidden_states + position_embeddings

        if self.block_position_encoding:
            block_position_embeddings = self.block_position_embeddings(block_position_ids)
            hidden_states = hidden_states + block_position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        new_caches = [hidden_states.detach()]
        for i, layer in enumerate(self.layers):
            mem_i = cache[i] if cache is not None else None
            has_gradient = not hidden_states.stop_gradient
            if self.enable_recompute and has_gradient:
                # TODO Should the attention_mask be added, it seems missing in original application.
                hidden_states = self.recompute_training(layer, hidden_states, attention_mask, cache=mem_i)
            else:
                hidden_states = layer(hidden_states, attention_mask, cache=mem_i)

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            new_caches.append(hidden_states.detach())

        output = self.final_layernorm(hidden_states)
        new_caches = self.update_memories(new_caches, cache)

        if not return_dict:
            return (output, new_caches)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output,
            past_key_values=new_caches,
        )

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


class GLMPretrainedModel(PretrainedModel):
    """
    An abstarct class for pretrained GLM models. It provides GLM related
    `model_config_file`, `resource_file_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    base_model_prefix = "glm"
    config_class = GLMConfig
    model_config_file = CONFIG_NAME
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_init_configuration = GLM_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = GLM_PRETRAINED_RESOURCE_FILES_MAP

    @classmethod
    def _get_name_mappings(cls, config):
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["word_embeddings.weight", "word_embeddings.weight"],
            ["transformer.position_embeddings.weight", "transformer.position_embeddings.weight"],
            ["transformer.block_position_embeddings.weight", "transformer.block_position_embeddings.weight"],
            ["transformer.final_layernorm.weight", "transformer.final_layernorm.weight"],
            ["transformer.final_layernorm.bias", "transformer.final_layernorm.bias"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = []
            transpose_names = [
                "attention.query_key_value.weight",
                "attention.dense.weight",
                "mlp.dense_h_to_4h.weight",
                "mlp.dense_4h_to_h.weight",
            ]
            mapping_names = [
                "attention.query_key_value.bias",
                "input_layernorm.weight",
                "input_layernorm.bias",
                "attention.dense.bias",
                "post_attention_layernorm.weight",
                "post_attention_layernorm.bias",
                "mlp.dense_h_to_4h.bias",
                "mlp.dense_4h_to_h.bias",
            ]
            for name in mapping_names:
                layer_mappings.append(
                    [f"transformer.layers.{layer_index}.{name}", f"transformer.layers.{layer_index}.{name}"]
                )
            for name in transpose_names:
                layer_mappings.append(
                    [
                        f"transformer.layers.{layer_index}.{name}",
                        f"transformer.layers.{layer_index}.{name}",
                        "transpose",
                    ]
                )

            model_mappings.extend(layer_mappings)

        if cls.__name__ != "GLMModel":
            for mapping in model_mappings:
                mapping[1] = "glm." + mapping[1]

        mappings = [StateDictNameMapping(*mapping) for mapping in model_mappings]
        return mappings

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, nn.Linear):
            std = self.config.initializer_range
            # TODO: initialization for glm-515m
            # if self.config.use_scaled_init_for_output_weights and _is_output_dense(layer):
            #     std = self.config.initializer_range / math.sqrt(2.0 * self.config.num_layers)
            normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                zeros_(layer.bias)
        elif isinstance(layer, nn.Embedding):
            normal_(layer.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(layer, nn.LayerNorm):
            ones_(layer.weight)
            zeros_(layer.bias)


@register_base_model
class GLMModel(GLMPretrainedModel):
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

    def __init__(self, config: GLMConfig):
        super(GLMModel, self).__init__(config)
        self.config = config
        self.output_predict = config.output_predict
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
        )
        self.transformer = GLMStack(config)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,
        attention_mask: Tensor = None,
        cache: Tensor = None,
        return_dict: bool = None,
    ):
        batch_size = input_ids.shape[0]
        word_embeddings = self.word_embeddings(input_ids)
        input_shape = input_ids.shape

        if position_ids is None:
            position_ids = paddle.arange(0, input_shape[-1], dtype="int64")
            block_position_ids = paddle.zeros(input_shape[-1:], dtype="int64")
            position_ids = paddle.stack([position_ids, block_position_ids], axis=0).unsqueeze(0)

        if attention_mask is None:
            attention_mask = paddle.zeros([batch_size])

        output = self.transformer(word_embeddings, position_ids, attention_mask, cache, return_dict)
        if return_dict:
            logits = output.last_hidden_state
        else:
            logits = output[0] if isinstance(output, tuple) else output

        if self.output_predict:
            logits = F.linear(logits, self.word_embeddings.weight.T)

        if not return_dict:
            if isinstance(output, tuple):
                return tuple([logits] + [v for v in output[1:]])
            else:
                return logits

        output.last_hidden_state = logits

        return output


class GLMForMultipleChoice(GLMPretrainedModel):
    """
    GLM Model transformer for multiple choice classification
    """

    def __init__(self, config: GLMConfig):
        super(GLMForMultipleChoice, self).__init__(config)
        self.glm = GLMModel(config)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,
        attention_mask: Tensor = None,
        choice_ids: Tensor = None,
        choice_indices: Tensor = None,
        labels: Tensor = None,
        return_dict: bool = None,
    ):
        model_output = self.glm(input_ids, position_ids, attention_mask, return_dict=return_dict)
        lm_logits = model_output.last_hidden_state if return_dict else model_output
        lm_logits = lm_logits[0] if isinstance(lm_logits, tuple) else lm_logits
        log_probs = []
        for output, choices, choice_index in zip(F.log_softmax(lm_logits, axis=-1), choice_ids, choice_indices):
            log_probs_single = []
            for choice, choice_target_id in zip(choices, choice_index):
                log_probs_single.append(output[choice_target_id, choice].sum()[0])
            log_probs.append(paddle.stack(log_probs_single))
        log_probs = paddle.stack(log_probs).squeeze(2)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(log_probs, labels)

        if not return_dict:
            output = (log_probs, lm_logits)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=log_probs,
            hidden_states=lm_logits,
        )


class GLMForConditionalGeneration(GLMPretrainedModel):
    """
    GLM Model transformer with a `language modeling` head on top.
    """

    def __init__(self, config: GLMConfig):
        super(GLMForConditionalGeneration, self).__init__(config)
        self.glm = GLMModel(config)
        self.apply(self.init_weights)

    def _reorder_cache(self, cache, beam_index):
        # Speedy decoding is disabled and no reorder is needed if decoder cache is not given.
        if cache is None:
            return None

        reordered_decoder_cache = ()
        for layer_cache_states in cache:
            # Get correct batch index from layer cache batch dimension
            reordered_decoder_cache = reordered_decoder_cache + (layer_cache_states.index_select(0, beam_index),)
        return reordered_decoder_cache

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        position_ids: Tensor = None,
        attention_mask: Tensor = None,
        cache: Tensor = None,
        **kwargs
    ):
        seq_length = input_ids.shape[1]
        if cache:
            if position_ids is not None:
                position_ids = position_ids[:, :, seq_length - 1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, seq_length - 1, :seq_length].unsqueeze(-2)
            input_ids = input_ids[:, -1].unsqueeze(-1)
        else:
            if position_ids is not None:
                position_ids = position_ids[:, :, :seq_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :seq_length, :seq_length]
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache": cache,
        }

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Tensor = None,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        cache: Tensor = None,
        return_dict: bool = None,
        loss_mask: Tensor = None,
    ):
        model_output = self.glm(input_ids, position_ids, attention_mask, cache=cache, return_dict=return_dict)
        if return_dict:
            lm_logits, cache = model_output.last_hidden_state, model_output.past_key_values
        else:
            lm_logits, cache = model_output

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                lm_logits.reshape([-1, lm_logits.shape[-1]]),
                labels.reshape([-1]),
                reduction="none",
            )
            label_smoothing = getattr(self.config, "label_smoothing", 0)
            if label_smoothing > 0:
                smooth_loss = (-F.log_softmax(lm_logits, axis=-1) / lm_logits.shape[2]).sum(axis=-1)
                loss = (1 - label_smoothing) * loss + label_smoothing * smooth_loss
            if loss_mask is not None:
                loss_mask = loss_mask.reshape([-1])
                loss = paddle.sum(loss.reshape([-1]) * loss_mask) / paddle.sum(loss_mask)

        if not return_dict:
            output = (lm_logits, cache)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(loss=loss, logits=lm_logits, past_key_values=cache)
