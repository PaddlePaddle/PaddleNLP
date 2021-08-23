# coding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" Paddle2.x T5 model. """

import copy
import logging
import math

import numpy as np
import paddle

import paddle.nn as nn
import paddle.nn.functional as F

from ..model_utils import PretrainedModel, register_base_model
from ..nezha.modeling import ACT2FN

from .utils import (BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions,
                    Seq2SeqLMOutput, Seq2SeqModelOutput, ModelOutput, Config)
from ..generation_utils import BeamSearchScorer

logger = logging.getLogger(__name__)


def finfo(dtype):
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


class T5LayerNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = paddle.pow(hidden_states.astype(paddle.float32), 2).mean(
            axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance +
                                                     self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == paddle.float16:
            hidden_states = hidden_states.astype(paddle.float16)
        return self.weight * hidden_states


class T5DenseReluDense(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias_attr=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias_attr=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias_attr=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias_attr=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias_attr=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Layer):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Layer):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias_attr=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                relative_position > 0).astype(paddle.int64) * num_buckets
            relative_position = paddle.abs(relative_position)
        else:
            relative_position = -paddle.minimum(
                relative_position, paddle.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (paddle.log(
            relative_position.astype(paddle.get_default_dtype()) /
            max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)).astype(paddle.int64)
        relative_postion_if_large = paddle.minimum(
            relative_postion_if_large,
            paddle.full_like(relative_postion_if_large, num_buckets - 1), )

        relative_buckets += paddle.where(is_small, relative_position,
                                         relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = paddle.arange(query_length).unsqueeze(-1)
        memory_position = paddle.arange(key_length).unsqueeze(0)
        relative_position = (memory_position - context_position
                             )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets, )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.transpose(perm=[2, 0, 1]).unsqueeze(
            0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            query_length=None,
            use_cache=False,
            output_attentions=False, ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (past_key_value[0].shape[2]
                                if query_length is None else query_length)

        key_length = (real_seq_length if key_value_states is None else
                      key_value_states.shape[1])

        def shape(states):
            """projection"""
            return states.reshape(
                shape=[batch_size, -1, self.n_heads,
                       self.key_value_proj_dim]).transpose(perm=[0, 2, 1, 3])

        def unshape(states):
            """reshape"""
            return states.transpose(perm=[0, 2, 1, 3]).reshape(
                shape=[batch_size, -1, self.inner_dim])

        def project(hidden_states, proj_layer, key_value_states,
                    past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = paddle.concat(
                        [past_key_value, hidden_states], axis=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(
            hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None, )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None, )

        # compute scores
        scores = paddle.matmul(query_states, key_states, transpose_y=True)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = paddle.zeros(
                    shape=(1, self.n_heads, real_seq_length, key_length),
                    dtype=scores.dtype, )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.shape[1]:, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = F.softmax(
            scores.astype(paddle.float32), axis=-1).astype(
                scores.dtype)  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.
            training)  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(paddle.matmul(
            attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = ((key_states, value_states)
                                   if (self.is_decoder and use_cache) else None)
        outputs = (attn_output, ) + (present_key_value_state, ) + (
            position_bias, )

        if output_attentions:
            outputs = outputs + (attn_weights, )
        return outputs


class T5LayerSelfAttention(nn.Layer):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False, ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions, )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,
                   ) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            key_value_states,
            attention_mask=None,
            position_bias=None,
            past_key_value=None,
            use_cache=False,
            query_length=None,
            output_attentions=False, ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions, )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,
                   ) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Layer):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.LayerList()
        self.layer.append(
            T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False, ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states")

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions, )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[
            2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == paddle.float16 and paddle.isinf(
                hidden_states).any():
            # TODO finfo
            clamp_value = finfo(hidden_states.dtype).max - 1000
            hidden_states = paddle.clip(
                hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions, )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if (hidden_states.dtype == paddle.float16 and
                    paddle.isinf(hidden_states).any()):
                # TODO
                clamp_value = finfo(hidden_states.dtype).max - 1000
                hidden_states = paddle.clip(
                    hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = (
                    present_key_value_state + cross_attention_outputs[1])

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == paddle.float16 and paddle.isinf(
                hidden_states).any():
            # TODO
            clamp_value = finfo(hidden_states.dtype).max - 1000
            hidden_states = paddle.clip(
                hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states, )

        if use_cache:
            outputs = outputs + (present_key_value_state, ) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5PreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    base_model_prefix = "t5"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "t5-small": {
            "return_dict": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "tie_word_embeddings": True,
            "is_encoder_decoder": True,
            "is_decoder": False,
            "pad_token_id": 0,
            "decoder_start_token_id": 0,
            "eos_token_id": 1,
            "vocab_size": 32128,
            "d_model": 512,
            "d_kv": 64,
            "d_ff": 2048,
            "num_layers": 6,
            "num_decoder_layers": 6,
            "num_heads": 8,
            "relative_attention_num_buckets": 32,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "initializer_factor": 1.0,
            "feed_forward_proj": "relu",
            "use_cache": True,
            "use_return_dict": True,
        },
        "t5-base": {
            "return_dict": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "tie_word_embeddings": True,
            "is_encoder_decoder": True,
            "is_decoder": False,
            "pad_token_id": 0,
            "decoder_start_token_id": 0,
            "eos_token_id": 1,
            "vocab_size": 32128,
            "d_model": 768,
            "d_kv": 64,
            "d_ff": 3072,
            "num_layers": 12,
            "num_decoder_layers": 12,
            "num_heads": 12,
            "relative_attention_num_buckets": 32,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "initializer_factor": 1.0,
            "feed_forward_proj": "relu",
            "use_cache": True,
            "use_return_dict": True
        },
        "t5-large": {
            "return_dict": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "tie_word_embeddings": True,
            "is_encoder_decoder": True,
            "is_decoder": False,
            "pad_token_id": 0,
            "decoder_start_token_id": 0,
            "eos_token_id": 1,
            "vocab_size": 32128,
            "d_model": 1024,
            "d_kv": 64,
            "d_ff": 4096,
            "num_layers": 24,
            "num_decoder_layers": 24,
            "num_heads": 16,
            "relative_attention_num_buckets": 32,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06,
            "initializer_factor": 1.0,
            "feed_forward_proj": "relu",
            "use_cache": True,
            "use_return_dict": True
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "t5-small":
            "https://huggingface.co/junnyu/t5/resolve/main/t5/t5-small/model_state.pdparams",
            "t5-base":
            "https://huggingface.co/junnyu/t5/resolve/main/t5/t5-base/model_state.pdparams",
            "t5-large":
            "https://huggingface.co/junnyu/t5/resolve/main/t5/t5-large/model_state.pdparams"
        }
    }

    @property
    def dummy_inputs(self):
        DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
        DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]
        input_ids = paddle.to_tensor(DUMMY_INPUTS, dtype=paddle.int64)
        input_mask = paddle.to_tensor(DUMMY_MASK, dtype=paddle.int64)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def init_weights(self):
        """
        Initializes and tie weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        """Initialize the weights"""
        factor = (self.pd_config.initializer_factor
                  )  # Used for testing weights initialization
        if isinstance(layer, T5LayerNorm):
            layer.weight.set_value(paddle.ones_like(layer.weight) * factor)
        elif isinstance(layer, T5Model):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            layer.shared.weight.set_value(
                paddle.normal(
                    mean=0.0, std=factor * 1.0,
                    shape=layer.shared.weight.shape))
        elif isinstance(layer, (T5ForConditionalGeneration, T5EncoderModel)):
            layer.t5.shared.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * 1.0,
                    shape=layer.t5.shared.weight.shape))

        elif isinstance(layer, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            layer.wi.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((self.pd_config.d_model)**-0.5),
                    shape=layer.wi.weight.shape))

            if hasattr(layer.wi, "bias") and layer.wi.bias is not None:
                layer.wi.bias.set_value(paddle.zeros_like(layer.wi.bias))

            layer.wo.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((self.pd_config.d_ff)**-0.5),
                    shape=layer.wo.weight.shape))

            if hasattr(layer.wo, "bias") and layer.wo.bias is not None:
                layer.wo.bias.set_value(paddle.zeros_like(layer.wo.bias))

        elif isinstance(layer, T5DenseGatedGeluDense):
            layer.wi_0.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((self.pd_config.d_model)**-0.5),
                    shape=layer.wi_0.weight.shape))
            if hasattr(layer.wi_0, "bias") and layer.wi_0.bias is not None:
                layer.wi_0.bias.set_value(paddle.zeros_like(layer.wi_0.bias))

            layer.wi_1.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((self.pd_config.d_model)**-0.5),
                    shape=layer.wi_1.weight.shape))
            if hasattr(layer.wi_1, "bias") and layer.wi_1.bias is not None:
                layer.wi_1.bias.set_value(paddle.zeros_like(layer.wi_1.bias))

            layer.wo.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((self.pd_config.d_ff)**-0.5),
                    shape=layer.wo.weight.shape))

            if hasattr(layer.wo, "bias") and layer.wo.bias is not None:
                layer.wo.bias.set_value(paddle.zeros_like(layer.wo.bias))
        elif isinstance(layer, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.pd_config.d_model
            key_value_proj_dim = self.pd_config.d_kv
            n_heads = self.pd_config.num_heads

            layer.q.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((d_model * key_value_proj_dim)**-0.5),
                    shape=layer.q.weight.shape))

            layer.k.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * (d_model**-0.5),
                    shape=layer.k.weight.shape))

            layer.v.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * (d_model**-0.5),
                    shape=layer.v.weight.shape))

            layer.o.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=factor * ((n_heads * key_value_proj_dim)**-0.5),
                    shape=layer.o.weight.shape))

            if layer.has_relative_attention_bias:
                layer.relative_attention_bias.weight.set_value(
                    paddle.normal(
                        mean=0.0,
                        std=factor * ((d_model)**-0.5),
                        shape=layer.relative_attention_bias.weight.shape))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.pd_config.decoder_start_token_id
        pad_token_id = self.pd_config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.pd_config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = paddle.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        assert (pad_token_id is not None
                ), "self.model.pd_config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = paddle.where(
            shifted_input_ids == -100,
            paddle.to_tensor(
                pad_token_id, dtype=shifted_input_ids.dtype),
            shifted_input_ids)

        assert paddle.all(shifted_input_ids >= 0).item(
        ), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, pd_config, embed_tokens=None):
        super().__init__()
        self.pd_config = pd_config
        self.embed_tokens = embed_tokens
        self.is_decoder = pd_config.is_decoder

        self.block = nn.LayerList([
            T5Block(
                pd_config, has_relative_attention_bias=bool(i == 0))
            for i in range(pd_config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(
            pd_config.d_model, eps=pd_config.layer_norm_epsilon)
        self.dropout = nn.Dropout(pd_config.dropout_rate)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):

        use_cache = use_cache if use_cache is not None else self.pd_config.use_cache
        output_attentions = (output_attentions if output_attentions is not None
                             else self.pd_config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.pd_config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else
                       self.pd_config.use_return_dict)

        input_shape = input_ids.shape
        input_ids = input_ids.reshape(shape=[-1, input_shape[-1]])

        inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (past_key_values[0][0].shape[2] + seq_length
                           if past_key_values is not None else seq_length)

        if use_cache is True:
            assert (
                self.is_decoder
            ), f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = paddle.ones(shape=[batch_size, mask_seq_length])
        if (self.is_decoder and encoder_attention_mask is None and
                encoder_hidden_states is not None):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = paddle.ones(
                [batch_size, encoder_seq_length], dtype=paddle.int64)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(shape=encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and
                                      self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module,
                past_key_value) in enumerate(zip(self.block, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions, )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None, ) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state, )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3], )
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (
                        layer_outputs[5], )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v
                         for v in [
                             hidden_states,
                             present_key_value_states,
                             all_hidden_states,
                             all_attentions,
                             all_cross_attentions,
                         ] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions, )

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.ndim == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.pd_config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = paddle.arange(seq_length)
                causal_mask = paddle.tile(
                    seq_ids.unsqueeze(axis=[0, 1]),
                    [batch_size, seq_length, 1]) <= seq_ids.unsqueeze(
                        axis=[0, 2])
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.astype(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[
                        1] - causal_mask.shape[1]
                    causal_mask = paddle.concat(
                        [
                            paddle.ones(
                                [batch_size, seq_length, prefix_seq_len],
                                dtype=causal_mask.dtype, ),
                            causal_mask,
                        ],
                        axis=-1, )

                extended_attention_mask = causal_mask.unsqueeze(
                    1) * attention_mask.unsqueeze([1, 2])
            else:
                extended_attention_mask = attention_mask.unsqueeze([1, 2])
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.astype(self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.ndim == 3:
            encoder_extended_attention_mask = encoder_attention_mask.unsqueeze(
                1)
        if encoder_attention_mask.ndim == 2:
            encoder_extended_attention_mask = encoder_attention_mask.unsqueeze(
                [1, 2])
        encoder_extended_attention_mask = encoder_extended_attention_mask.astype(
            self.dtype)  # fp16 compatibility

        if self.dtype == paddle.float16:
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype == paddle.float32:
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{self.dtype} not recognized. `dtype` should be set to either `paddle.float32` or `paddle.float16`"
            )

        return encoder_extended_attention_mask


@register_base_model
class T5Model(T5PreTrainedModel):
    def __init__(self, **kwargs):
        super().__init__()
        pd_config = Config(**kwargs)
        self.pd_config = pd_config

        self.shared = nn.Embedding(pd_config.vocab_size, pd_config.d_model)

        encoder_config = copy.deepcopy(pd_config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(pd_config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = pd_config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):

        use_cache = use_cache if use_cache is not None else self.pd_config.use_cache
        return_dict = (return_dict if return_dict is not None else
                       self.pd_config.use_return_dict)

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict, )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1]
                if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2 else None, )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions, )


class T5ForConditionalGeneration(T5PreTrainedModel):
    def __init__(self, t5):
        super().__init__()
        self.t5 = t5
        self.pd_config = t5.pd_config
        if not self.pd_config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                self.pd_config.d_model,
                self.pd_config.vocab_size,
                bias_attr=False)

        self.eos_token_id = self.pd_config.eos_token_id
        self.pad_token_id = self.pd_config.pad_token_id
        self.init_weights()

    def get_input_embeddings(self):
        return self.t5.shared

    def set_input_embeddings(self, new_embeddings):
        self.t5.shared = new_embeddings
        self.t5.encoder.set_input_embeddings(new_embeddings)
        self.t5.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        if not self.pd_config.tie_word_embeddings:
            return self.t5.shared
        return self.lm_head

    def get_encoder(self):
        return self.t5.encoder

    def get_decoder(self):
        return self.t5.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):

        use_cache = use_cache if use_cache is not None else self.pd_config.use_cache
        return_dict = (return_dict if return_dict is not None else
                       self.pd_config.use_return_dict)

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict, )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1]
                if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2 else None, )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

        # Decode
        decoder_outputs = self.t5.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        sequence_output = decoder_outputs[0]

        if self.pd_config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.pd_config.d_model**-0.5)
            lm_logits = paddle.matmul(
                sequence_output, self.t5.shared.weight, transpose_y=True)
        else:
            lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.reshape(shape=[-1, lm_logits.shape[-1]]),
                labels.flatten())
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits, ) + decoder_outputs[1:] + encoder_outputs
            return ((loss, ) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions, )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs, ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(beam_idx), )

            assert reordered_layer_past_states[0].shape == layer_past_states[
                0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states, )
        return reordered_decoder_past

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids,
                                                       model_kwargs):
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith(
                    "cross_attn"))
            }
            model_kwargs["encoder_outputs"] = encoder(
                input_ids, return_dict=True, **encoder_kwargs)

        return model_kwargs

    def _get_decoder_start_token_id(self,
                                    decoder_start_token_id: int=None,
                                    bos_token_id: int=None) -> int:
        decoder_start_token_id = (decoder_start_token_id
                                  if decoder_start_token_id is not None else
                                  self.pd_config.decoder_start_token_id)
        bos_token_id = bos_token_id if bos_token_id is not None else self.pd_config.get(
            bos_token_id, None)

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif (hasattr(self.pd_config, "decoder") and
              hasattr(self.pd_config.decoder, "decoder_start_token_id") and
              self.pd_config.decoder.decoder_start_token_id is not None):
            return self.pd_config.decoder.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif (hasattr(self.pd_config, "decoder") and
              hasattr(self.pd_config.decoder, "bos_token_id") and
              self.pd_config.decoder.bos_token_id is not None):
            return self.pd_config.decoder.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    def _prepare_decoder_input_ids_for_generation(
            self,
            input_ids,
            decoder_start_token_id: int=None,
            bos_token_id: int=None):
        decoder_start_token_id = self._get_decoder_start_token_id(
            decoder_start_token_id, bos_token_id)
        decoder_input_ids = (paddle.ones(
            shape=(input_ids.shape[0], 1),
            dtype=paddle.int64) * decoder_start_token_id)
        return decoder_input_ids

    @paddle.no_grad()
    def generate(self,
                 input_ids=None,
                 max_length=20,
                 min_length=0,
                 decode_strategy='greedy_search',
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
                 num_beams=1,
                 length_penalty=1.0,
                 early_stopping=False,
                 bos_token_id=None,
                 eos_token_id=None,
                 pad_token_id=None,
                 num_return_sequences=1,
                 use_cache=True,
                 **model_kwargs):

        # params check
        bos_token_id = bos_token_id if bos_token_id is not None else getattr(
            self, 'bos_token_id', None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(
            self, 'eos_token_id', None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(
            self, 'pad_token_id', None)

        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs[
                "attention_mask"] = self.prepare_attention_mask_for_generation(
                    input_ids, pad_token_id, eos_token_id)

        if pad_token_id is None and eos_token_id is not None:
            print("Setting `pad_token_id` to `eos_token_id`:{} for "
                  "open-end generation.".format(eos_token_id))
            pad_token_id = eos_token_id

        # TODO Add relevant processing for encoder_decoder model.
        if self.pd_config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids,
                    decoder_start_token_id=self.pd_config.
                    decoder_start_token_id,
                    bos_token_id=bos_token_id)
            if "encoder_outputs" not in model_kwargs or not isinstance(
                    model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError(
                    "Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`."
                )

        model_kwargs["use_cache"] = use_cache

        max_length += input_ids.shape[-1]

        logits_processors = self.get_logits_processor(min_length, eos_token_id)

        if decode_strategy == 'greedy_search':
            if num_return_sequences > 1:
                raise ValueError(
                    "`num_return_sequences` has to be 1, but is {} "
                    "when doing greedy search.".format(num_return_sequences))

            return self.greedy_search(input_ids, logits_processors, max_length,
                                      pad_token_id, eos_token_id,
                                      **model_kwargs)

        elif decode_strategy == 'sampling':
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs)

            return self.sample(input_ids, logits_processors, max_length,
                               pad_token_id, eos_token_id, top_k, top_p,
                               temperature, **model_kwargs)

        elif decode_strategy == 'beam_search':
            batch_size = input_ids.shape[0]
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to "
                    "`num_beams`. But received `num_return_sequences` is {}, "
                    "`num_beams` is {}".format(num_return_sequences, num_beams))
            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` has to be bigger than 1. But received "
                    "`num_beams` is {}. If `num_beams` is 1, `decode_strategy` "
                    "should be 'greedy_search'".format(num_beams))

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences)

            input_ids, model_kwargs = self.expand_inputs_for_generation(
                input_ids, expand_size=num_beams, **model_kwargs)

            return self.beam_search(input_ids, beam_scorer, logits_processors,
                                    max_length, pad_token_id, eos_token_id,
                                    **model_kwargs)

        else:
            raise ValueError(
                '`decode_strategy` must be one of "greedy_search", "sampling" '
                'and "beam_search".')

    #### for t5
    @staticmethod
    def expand_inputs_for_generation(input_ids,
                                     expand_size,
                                     attention_mask=None,
                                     encoder_outputs=None,
                                     **model_kwargs):
        index = paddle.tile(
            paddle.arange(input_ids.shape[0]).unsqueeze(-1),
            [1, expand_size]).reshape([-1])

        input_ids = paddle.index_select(input_ids, index)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = paddle.index_select(attention_mask,
                                                                 index)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.index_select(token_type_ids,
                                                                 index)

        assert encoder_outputs is not None
        encoder_outputs["last_hidden_state"] = paddle.index_select(
            encoder_outputs.last_hidden_state, index)
        model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs

    def beam_search(self, input_ids, beam_scorer, logits_processors, max_length,
                    pad_token_id, eos_token_id, **model_kwargs):
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size)

        beam_scores = paddle.zeros(
            (batch_size, num_beams), dtype=paddle.get_default_dtype())
        beam_scores[:, 1:] = -1e9
        beam_scores = paddle.reshape(beam_scores, [-1])

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs, return_dict=True)
            logits = outputs.logits
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # beam search
            # [batch_size * num_beams, vocab_size]
            next_scores = F.softmax(logits)
            next_scores = paddle.log(next_scores)

            next_scores = next_scores + beam_scores.unsqueeze(-1)
            # reshape for beam search
            vocab_size = next_scores.shape[-1]
            next_scores = next_scores.reshape(
                [batch_size, num_beams * vocab_size])

            next_scores, next_tokens = paddle.topk(
                next_scores, 2 * num_beams, axis=1)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                origin_len=origin_len,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id, )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            cur_len += 1
            input_ids = paddle.concat(
                [
                    paddle.index_select(input_ids, beam_idx),
                    beam_next_tokens.unsqueeze(-1)
                ],
                axis=-1)

            if beam_scorer.is_done:
                break
            model_kwargs = self.update_model_kwargs_for_generation(outputs,
                                                                   model_kwargs)

            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"],
                                                           beam_idx)

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id)
        return pred_ids, scores

    def greedy_search(self, input_ids, logits_processors, max_length,
                      pad_token_id, eos_token_id, **model_kwargs):

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype='bool')
        scores = paddle.full(
            [batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        while cur_len < max_length:
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids,
                                                              **model_kwargs)
            outputs = self(**model_inputs, return_dict=True)

            logits = outputs.logits
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # greedy
            probs = F.softmax(logits)
            probs = paddle.log(probs)
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens,
                                           paddle.full_like(next_tokens,
                                                            pad_token_id))

            scores = self.update_scores_for_generation(
                scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(
                    unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break

            model_kwargs = self.update_model_kwargs_for_generation(outputs,
                                                                   model_kwargs)
        return input_ids, scores

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id,
                                              eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id))
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids == pad_token_id
                              ).astype(paddle.get_default_dtype()) * -1e9
        else:
            attention_mask = paddle.zeros_like(
                input_ids, dtype=paddle.get_default_dtype())
        return attention_mask

    def update_model_kwargs_for_generation(self, outputs, model_kwargs):
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update attention mask
        if not self.pd_config.is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.cat([
                    attention_mask, paddle.ones(
                        shape=(attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype)
                ],
                                                            dim=-1)

        return model_kwargs


class T5EncoderModel(T5PreTrainedModel):
    def __init__(self, t5):
        super().__init__()
        self.t5 = t5
        del self.t5.decoder

        self.init_weights()

    def get_input_embeddings(self):
        return self.t5.shared

    def set_input_embeddings(self, new_embeddings):
        self.t5.shared = new_embeddings
        self.t5.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.t5.encoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):

        return_dict = (return_dict if return_dict is not None else
                       self.t5.pd_config.use_return_dict)

        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        return encoder_outputs
