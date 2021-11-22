# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" Modeling classes for LayoutXLM model."""

import copy
import math
import paddle
import paddle.nn as nn
import paddle.tensor as tensor
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.nn import CrossEntropyLoss

from .. import PretrainedModel, register_base_model

__all__ = [
    'LayoutLMModel',
    "LayoutLMPretrainedModel",
    "LayoutLMForTokenClassification",
    "LayoutLMForSequenceClassification",
]


class LayoutLMPooler(Layer):
    def __init__(self, hidden_size, pool_act='tanh'):
        super(LayoutLMPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == 'tanh':
            pooled_output = self.activation(pooled_output)
        return pooled_output


class LayoutLMEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):
        super(LayoutLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config["vocab_size"], config["hidden_size"], padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config["max_position_embeddings"], config["hidden_size"])
        # gry add for layoutlm
        self.x_position_embeddings = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"])
        self.y_position_embeddings = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"])
        self.h_position_embeddings = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"])
        self.w_position_embeddings = nn.Embedding(
            config["max_2d_position_embeddings"], config["coordinate_size"])
        # end of gry add for layoutlm
        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"],
                                                  config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(
            config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.register_buffer(
            "position_ids",
            paddle.arange(config["max_position_embeddings"]).expand((1, -1)))

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :,
                                                                        1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :,
                                                                        2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :,
                                                                        3])
        except IndexError as e:
            raise IndexError(
                "The :obj:`bbox`coordinate values should be within 0-1000 range."
            ) from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] -
                                                           bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] -
                                                           bbox[:, :, 0])

        spatial_position_embeddings = paddle.concat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            axis=-1, )
        return spatial_position_embeddings

    def forward(self,
                input_ids,
                bbox=None,
                token_type_ids=None,
                position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :,
                                                                        1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :,
                                                                        2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :,
                                                                        3])
        except IndexError as e:
            raise IndexError(
                "The :obj:`bbox`coordinate values should be within 0-1000 range."
            ) from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] -
                                                           bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] -
                                                           bbox[:, :, 0])

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            input_embedings + position_embeddings + left_position_embeddings +
            upper_position_embeddings + right_position_embeddings +
            lower_position_embeddings + h_position_embeddings +
            w_position_embeddings + token_type_embeddings)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutLMSelfOutput(nn.Layer):
    def __init__(self, config):
        super(LayoutLMSelfOutput, self).__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(
            config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutLMSelfAttention(nn.Layer):
    def __init__(self, config):
        super(LayoutLMSelfAttention, self).__init__()
        if config["hidden_size"] % config[
                "num_attention_heads"] != 0 and not hasattr(config,
                                                            "embedding_size"):
            raise ValueError(
                "The hidden size {} is not a multiple of the number of attention "
                "heads {}".format(config["hidden_size"], config[
                    "num_attention_heads"]))
        self.fast_qkv = config["fast_qkv"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = int(config["hidden_size"] /
                                       config["num_attention_heads"])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config["has_relative_attention_bias"]
        self.has_spatial_attention_bias = config["has_spatial_attention_bias"]

        if config["fast_qkv"]:
            self.qkv_linear = nn.Linear(
                config["hidden_size"], 3 * self.all_head_size, bias_attr=False)
            self.q_bias = self.create_parameter(
                shape=[1, 1, self.all_head_size],
                default_initializer=nn.initializer.Constant(0.0))
            self.v_bias = self.create_parameter(
                shape=[1, 1, self.all_head_size],
                default_initializer=nn.initializer.Constant(0.0))
        else:
            self.query = nn.Linear(config["hidden_size"], self.all_head_size)
            self.key = nn.Linear(config["hidden_size"], self.all_head_size)
            self.value = nn.Linear(config["hidden_size"], self.all_head_size)

        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads, self.attention_head_size
        ]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = paddle.chunk(qkv, 3, axis=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1, ) * (q.ndimension() - 1) + (-1, )
                q = q + self.q_bias.reshape(_sz)
                v = v + self.v_bias.vreshape(_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None, ):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        # [BSZ, NAT, L, L]
        attention_scores = paddle.matmul(query_layer,
                                         key_layer.transpose([0, 1, 3, 2]))
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        attention_scores = paddle.where(
            attention_mask.astype(paddle.bool).expand_as(attention_scores),
            paddle.ones_like(attention_scores) * float("-inf"),
            attention_scores)
        attention_probs = F.softmax(attention_scores, axis=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer, )
        return outputs


class LayoutLMAttention(nn.Layer):
    def __init__(self, config):
        super(LayoutLMAttention, self).__init__()
        self.self = LayoutLMSelfAttention(config)
        self.output = LayoutLMSelfOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None, ):

        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos, )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,
                   ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class LayoutLMEncoder(nn.Layer):
    def __init__(self, config):
        super(LayoutLMEncoder, self).__init__()
        self.config = config
        self.layer = nn.LayerList([
            LayoutLMLayer(config) for _ in range(config["num_hidden_layers"])
        ])

        self.has_relative_attention_bias = config["has_relative_attention_bias"]
        self.has_spatial_attention_bias = config["has_spatial_attention_bias"]

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config["rel_pos_bins"]
            self.max_rel_pos = config["max_rel_pos"]
            self.rel_pos_onehot_size = config["rel_pos_bins"]
            self.rel_pos_bias = nn.Linear(
                self.rel_pos_onehot_size,
                config["num_attention_heads"],
                bias_attr=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config["max_rel_2d_pos"]
            self.rel_2d_pos_bins = config["rel_2d_pos_bins"]
            self.rel_2d_pos_onehot_size = config["rel_2d_pos_bins"]
            self.rel_pos_x_bias = nn.Linear(
                self.rel_2d_pos_onehot_size,
                config["num_attention_heads"],
                bias_attr=False)
            self.rel_pos_y_bias = nn.Linear(
                self.rel_2d_pos_onehot_size,
                config["num_attention_heads"],
                bias_attr=False)

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos, )
        rel_pos = paddle.nn.functional.one_hot(
            rel_pos,
            num_classes=self.rel_pos_onehot_size).astype(hidden_states.dtype)
        rel_pos = self.rel_pos_bias(rel_pos).transpose([0, 3, 1, 2])
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(
            -2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(
            -2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos, )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos, )
        rel_pos_x = F.one_hot(
            rel_pos_x,
            num_classes=self.rel_2d_pos_onehot_size).astype(hidden_states.dtype)
        rel_pos_y = F.one_hot(
            rel_pos_y,
            num_classes=self.rel_2d_pos_onehot_size).astype(hidden_states.dtype)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).transpose([0, 3, 1, 2])
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).transpose([0, 3, 1, 2])
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            output_attentions=False,
            output_hidden_states=False,
            bbox=None,
            position_ids=None, ):
        all_hidden_states = () if output_hidden_states else None

        rel_pos = self._cal_1d_pos_emb(
            hidden_states,
            position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(
            hidden_states, bbox) if self.has_spatial_attention_bias else None

        hidden_save = dict()
        hidden_save["input_hidden_states"] = hidden_states

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[
                i] if past_key_values is not None else None

            # gradient_checkpointing is set as False here so we remove some codes here
            hidden_save["input_attention_mask"] = attention_mask
            hidden_save["input_layer_head_mask"] = layer_head_mask
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos, )

            hidden_states = layer_outputs[0]

            hidden_save["{}_data".format(i)] = hidden_states

        return hidden_states,


class LayoutLMIntermediate(nn.Layer):
    def __init__(self, config):
        super(LayoutLMIntermediate, self).__init__()
        self.dense = nn.Linear(config["hidden_size"],
                               config["intermediate_size"])
        if config["hidden_act"] == "gelu":
            self.intermediate_act_fn = nn.GELU()
        else:
            assert False, "hidden_act is set as: {}, please check it..".format(
                config["hidden_act"])

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LayoutLMOutput(nn.Layer):
    def __init__(self, config):
        super(LayoutLMOutput, self).__init__()
        self.dense = nn.Linear(config["intermediate_size"],
                               config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(
            config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutLMLayer(nn.Layer):
    def __init__(self, config):
        super(LayoutLMLayer, self).__init__()
        # since chunk_size_feed_forward is 0 as default, no chunk is needed here.
        self.seq_len_dim = 1
        self.attention = LayoutLMAttention(config)
        self.add_cross_attention = False  # default as false
        self.intermediate = LayoutLMIntermediate(config)
        self.output = LayoutLMOutput(config)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None, ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos, )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[
            1:]  # add self attentions if we output attention weights

        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output, ) + outputs

        return outputs


class LayoutLMPretrainedModel(PretrainedModel):
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "layoutlm-base-uncased": {
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 0,
            "coordinate_size": 128,
            "eos_token_id": 2,
            "fast_qkv": False,
            "gradient_checkpointing": False,
            "has_relative_attention_bias": False,
            "has_spatial_attention_bias": False,
            "has_visual_segment_embedding": True,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "image_feature_pool_shape": [7, 7, 256],
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_2d_position_embeddings": 1024,
            "max_position_embeddings": 514,
            "max_rel_2d_pos": 256,
            "max_rel_pos": 128,
            "model_type": "layoutlmv2",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "output_past": True,
            "pad_token_id": 1,
            "shape_size": 128,
            "rel_2d_pos_bins": 64,
            "rel_pos_bins": 32,
            "type_vocab_size": 1,
            "vocab_size": 250002,
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "layoutlm-base-uncased":
            "https://paddlenlp.bj.bcebos.com/models/transformers/layoutlm_base-uncased/model_state.pdparams",
        }
    }
    base_model_prefix = "layoutlm"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.pretrained_init_configuration[
                            "initializer_range"] if "initializer_range" in
                        self.pretrained_init_configuration else 0.02,
                        shape=layer.weight.shape))


@register_base_model
class LayoutLMModel(LayoutLMPretrainedModel):
    """
    The bare LayoutLM Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (`int`):
            Vocabulary size of the XLNet model. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling XLNetModel.
        hidden_size (`int`, optional):
            Dimensionality of the encoder layers and the pooler layer. Defaults to ``768``.
        num_hidden_layers (`int`, optional):
            Number of hidden layers in the Transformer encoder. Defaults to ``12``.
        num_attention_heads (`int`, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to ``12``.
        intermediate_size (`int`, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            Defaults to ``3072``.
        hidden_act (`str`, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to ``0.1``.
        attention_probs_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the pooler.
            Defaults to ``0.1``.
        initializer_range (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
    """

    def __init__(
            self,
            pool_act='tanh',
            **kwargs, ):
        super(LayoutLMModel, self).__init__()
        config = kwargs
        self.config = kwargs
        self.has_visual_segment_embedding = config[
            "has_visual_segment_embedding"]
        self.embeddings = LayoutXLMEmbeddings(config)

        self.visual = VisualBackbone(config)
        self.visual.stop_gradient = True
        self.visual_proj = nn.Linear(config["image_feature_pool_shape"][-1],
                                     config["hidden_size"])
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = self.create_parameter(
                shape=[config["hidden_size"], ], dtype=paddle.float32)
        self.visual_LayerNorm = nn.LayerNorm(
            config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.visual_dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.encoder = LayoutLMEncoder(config)
        self.pooler = LayoutLMPooler(config["hidden_size"], pool_act)

    def _calc_text_embeddings(self, input_ids, bbox, position_ids,
                              token_type_ids):
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(
            bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(
            token_type_ids)
        embeddings = words_embeddings + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        visual_embeddings = self.visual_proj(
            self.visual(image.astype(paddle.float32)))
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(
            bbox)
        embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def forward(self,
                input_ids=None,
                bbox=None,
                image=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                head_mask=None,
                output_hidden_states=None,
                output_attentions=None):
        input_shape = input_ids.shape

        visual_shape = list(input_shape)
        visual_shape[1] = self.config["image_feature_pool_shape"][
            0] * self.config["image_feature_pool_shape"][1]
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]

        visual_bbox_x = (paddle.arange(
            0,
            1000 * (self.config["image_feature_pool_shape"][1] + 1),
            1000,
            dtype=bbox.dtype, ) // self.config["image_feature_pool_shape"][1])
        visual_bbox_y = (paddle.arange(
            0,
            1000 * (self.config["image_feature_pool_shape"][0] + 1),
            1000,
            dtype=bbox.dtype, ) // self.config["image_feature_pool_shape"][0])

        expand_shape = self.config["image_feature_pool_shape"][0:2]

        visual_bbox = paddle.stack(
            [
                visual_bbox_x[:-1].expand(expand_shape),
                visual_bbox_y[:-1].expand(expand_shape[::-1]).transpose([1, 0]),
                visual_bbox_x[1:].expand(expand_shape),
                visual_bbox_y[1:].expand(expand_shape[::-1]).transpose([1, 0]),
            ],
            axis=-1, ).reshape([-1, bbox.shape[-1]])
        visual_bbox = visual_bbox.expand([final_shape[0], -1, -1])
        final_bbox = paddle.concat([bbox, visual_bbox], axis=1)

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)

        visual_attention_mask = paddle.ones(visual_shape)

        attention_mask = attention_mask.astype(visual_attention_mask.dtype)

        final_attention_mask = paddle.concat(
            [attention_mask, visual_attention_mask], axis=1)

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand_as(input_ids)

        visual_position_ids = paddle.arange(0, visual_shape[1]).expand(
            [input_shape[0], -1])
        final_position_ids = paddle.concat(
            [position_ids, visual_position_ids], axis=1)

        if bbox is None:
            bbox = paddle.zeros(input_shape + [4])

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids, )

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids, )
        final_emb = paddle.concat([text_layout_emb, visual_emb], axis=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                    -1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config["num_hidden_layers"],
                                             -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config["num_hidden_layers"]

        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class LayoutLMForTokenClassification(LayoutLMPretrainedModel):
    def __init__(self, layoutlm, num_classes=2, dropout=None):
        super(LayoutLMForTokenClassification, self).__init__()
        self.num_classes = num_classes
        if isinstance(layoutlm, dict):
            self.layoutlm = LayoutLMModel(**layoutxlm)
        else:
            self.layoutlm = layoutlm
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.layoutlm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.layoutlm.config["hidden_size"],
                                    num_classes)
        self.classifier.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def forward(
            self,
            input_ids=None,
            bbox=None,
            image=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None, ):
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask, )
        seq_length = input_ids.shape[1]
        # sequence out and image out
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[
            0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = logits,

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.reshape([-1, ]) == 1
                active_logits = logits.reshape(
                    [-1, self.num_classes])[active_loss]
                active_labels = labels.reshape([-1, ])[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.reshape([-1, self.num_classes]),
                    labels.reshape([-1, ]))

            outputs = (loss, ) + outputs

        return outputs


class LayoutLMForSequenceClassification(LayoutLMPretrainedModel):
    def __init__(self, layoutlm, num_classes=2, dropout=None):
        super(LayoutLMForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        if isinstance(layoutlm, dict):
            self.layoutlm = LayoutLMModel(**layoutlm)
        else:
            self.layoutlm = layoutlm
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.layoutlm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.layoutlm.config["hidden_size"],
                                    num_classes)
        self.classifier.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def forward(
            self,
            input_ids=None,
            bbox=None,
            image=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None, ):
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask, )
        seq_length = input_ids.shape[1]
        # sequence out and image out
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[
            0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = logits,

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.reshape([-1, ]) == 1
                active_logits = logits.reshape(
                    [-1, self.num_classes])[active_loss]
                active_labels = labels.reshape([-1, ])[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.reshape([-1, self.num_classes]),
                    labels.reshape([-1, ]))

            outputs = (loss, ) + outputs

        return outputs


class LayoutLMPredictionHead(Layer):
    """
    Bert Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(LayoutLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[vocab_size, hidden_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states


class LayoutLMPretrainingHeads(Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(LayoutLMPretrainingHeads, self).__init__()
        self.predictions = LayoutLMPredictionHead(hidden_size, vocab_size,
                                                  activation, embedding_weights)

    def forward(self, sequence_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores
