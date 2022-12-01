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

from paddlenlp.utils.log import logger
from .. import PretrainedModel, register_base_model
from .visual_backbone import build_resnet_fpn_backbone
from .visual_backbone import build_resnet_backbone
from .visual_backbone import read_config

__all__ = [
    "LayoutXLMModel",
    "LayoutXLMPretrainedModel",
    "LayoutXLMForTokenClassification",
    "LayoutXLMForSequenceClassification",
    "LayoutXLMForPretraining",
    "LayoutXLMForRelationExtraction",
    "LayoutXLMForQuestionAnswering",
]


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).astype(paddle.int64) * num_buckets
        n = paddle.abs(relative_position)
    else:
        n = paddle.max(-relative_position, paddle.zeros_like(relative_position))
    # Now n is in the range [0, inf)
    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        paddle.log(n.astype(paddle.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(paddle.int64)

    val_if_large = paddle.minimum(val_if_large, paddle.full_like(val_if_large, num_buckets - 1))

    ret += paddle.where(is_small, n, val_if_large)
    return ret


def token_featue_to_sequence_feature(input_ids, seq_length, sequence_output):
    """
    used to transform token feature into sequence feature by
    averaging all the token features of certain sequence
    """
    batches = input_ids.shape[0]
    for batch_id in range(batches):
        start_idx = -1
        for i in range(0, seq_length):
            if input_ids[batch_id, i] == 6:
                if start_idx > -1:
                    feature_block = sequence_output[batch_id, start_idx + 1 : i]
                    sequence_output[batch_id, start_idx] = paddle.mean(feature_block, axis=0)
                start_idx = i

            if input_ids[batch_id, i] == 1:
                feature_block = sequence_output[batch_id, start_idx + 1 : i]
                sequence_output[batch_id, start_idx] = paddle.mean(feature_block, axis=0)
                break

        if i == seq_length - 1:
            sequence_output[batch_id, start_idx] = paddle.mean(feature_block, axis=0)
        return


class LayoutXLMPooler(Layer):
    def __init__(self, hidden_size, with_pool):
        super(LayoutXLMPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.with_pool = with_pool

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.with_pool == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class LayoutXLMEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config):
        super(LayoutXLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"], padding_idx=0)
        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])

        self.x_position_embeddings = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.y_position_embeddings = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.h_position_embeddings = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])
        self.w_position_embeddings = nn.Embedding(config["max_2d_position_embeddings"], config["coordinate_size"])

        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.register_buffer("position_ids", paddle.arange(config["max_position_embeddings"]).expand((1, -1)))

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        spatial_position_embeddings = paddle.concat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            axis=-1,
        )
        return spatial_position_embeddings

    def forward(self, input_ids, bbox=None, token_type_ids=None, position_ids=None):
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
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            input_embedings
            + position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutXLMPretrainedModel(PretrainedModel):
    pretrained_init_configuration = {
        "layoutxlm-base-uncased": {
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
        },
        "vi-layoutxlm-base-uncased": {
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 0,
            "coordinate_size": 128,
            "eos_token_id": 2,
            "fast_qkv": False,
            "gradient_checkpointing": False,
            "has_relative_attention_bias": False,
            "has_spatial_attention_bias": False,
            "has_visual_segment_embedding": True,
            "use_visual_backbone": False,
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
        },
    }
    pretrained_resource_files_map = {
        "model_state": {
            "layoutxlm-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/layoutxlm_base/model_state.pdparams",
            "vi-layoutxlm-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/vi-layoutxlm-base-uncased/model_state.pdparams",
        }
    }
    base_model_prefix = "layoutxlm"

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.pretrained_init_configuration["initializer_range"]
                        if "initializer_range" in self.pretrained_init_configuration
                        else 0.02,
                        shape=layer.weight.shape,
                    )
                )


class LayoutXLMSelfOutput(nn.Layer):
    def __init__(self, config):
        super(LayoutXLMSelfOutput, self).__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutXLMSelfAttention(nn.Layer):
    def __init__(self, config):
        super(LayoutXLMSelfAttention, self).__init__()
        if config["hidden_size"] % config["num_attention_heads"] != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size {} is not a multiple of the number of attention "
                "heads {}".format(config["hidden_size"], config["num_attention_heads"])
            )
        self.fast_qkv = config["fast_qkv"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = int(config["hidden_size"] / config["num_attention_heads"])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config["has_relative_attention_bias"]
        self.has_spatial_attention_bias = config["has_spatial_attention_bias"]

        if config["fast_qkv"]:
            self.qkv_linear = nn.Linear(config["hidden_size"], 3 * self.all_head_size, bias_attr=False)
            self.q_bias = self.create_parameter(
                shape=[1, 1, self.all_head_size], default_initializer=nn.initializer.Constant(0.0)
            )
            self.v_bias = self.create_parameter(
                shape=[1, 1, self.all_head_size], default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.query = nn.Linear(config["hidden_size"], self.all_head_size)
            self.key = nn.Linear(config["hidden_size"], self.all_head_size)
            self.value = nn.Linear(config["hidden_size"], self.all_head_size)

        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])

    def transpose_for_scores(self, x):
        new_x_shape = list(x.shape[:-1]) + [self.num_attention_heads, self.attention_head_size]

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
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
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
        rel_2d_pos=None,
    ):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        # [BSZ, NAT, L, L]
        attention_scores = paddle.matmul(query_layer, key_layer.transpose([0, 1, 3, 2]))
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        bool_attention_mask = attention_mask.astype(paddle.bool)
        bool_attention_mask.stop_gradient = True
        attention_scores_shape = paddle.shape(attention_scores)
        attention_scores = paddle.where(
            bool_attention_mask.expand(attention_scores_shape),
            paddle.ones(attention_scores_shape) * float("-1e10"),
            attention_scores,
        )
        attention_probs = F.softmax(attention_scores, axis=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = list(context_layer.shape[:-2]) + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        if output_attentions:
            outputs = [context_layer, attention_probs]
        else:
            outputs = [context_layer]
        return outputs


class LayoutXLMAttention(nn.Layer):
    def __init__(self, config):
        super(LayoutXLMAttention, self).__init__()
        self.self = LayoutXLMSelfAttention(config)
        self.output = LayoutXLMSelfOutput(config)

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
        rel_2d_pos=None,
    ):

        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        if output_attentions:
            outputs = [
                attention_output,
            ] + self_outputs[1:]
        else:
            outputs = [attention_output]
        return outputs


class LayoutXLMEncoder(nn.Layer):
    def __init__(self, config):
        super(LayoutXLMEncoder, self).__init__()
        self.config = config
        self.layer = nn.LayerList([LayoutXLMLayer(config) for _ in range(config["num_hidden_layers"])])

        self.has_relative_attention_bias = config["has_relative_attention_bias"]
        self.has_spatial_attention_bias = config["has_spatial_attention_bias"]

        if self.has_relative_attention_bias:
            self.rel_pos_bins = config["rel_pos_bins"]
            self.max_rel_pos = config["max_rel_pos"]
            self.rel_pos_onehot_size = config["rel_pos_bins"]
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config["num_attention_heads"], bias_attr=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config["max_rel_2d_pos"]
            self.rel_2d_pos_bins = config["rel_2d_pos_bins"]
            self.rel_2d_pos_onehot_size = config["rel_2d_pos_bins"]
            self.rel_pos_x_bias = nn.Linear(
                self.rel_2d_pos_onehot_size, config["num_attention_heads"], bias_attr=False
            )
            self.rel_pos_y_bias = nn.Linear(
                self.rel_2d_pos_onehot_size, config["num_attention_heads"], bias_attr=False
            )

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = paddle.nn.functional.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).astype(
            hidden_states.dtype
        )
        rel_pos = self.rel_pos_bias(rel_pos).transpose([0, 3, 1, 2])
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).astype(hidden_states.dtype)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).astype(hidden_states.dtype)
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
        position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None

        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None

        hidden_save = dict()
        hidden_save["input_hidden_states"] = hidden_states

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

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
                rel_2d_pos=rel_2d_pos,
            )

            hidden_states = layer_outputs[0]

            hidden_save["{}_data".format(i)] = hidden_states

        return hidden_states, hidden_save


class LayoutXLMIntermediate(nn.Layer):
    def __init__(self, config):
        super(LayoutXLMIntermediate, self).__init__()
        self.dense = nn.Linear(config["hidden_size"], config["intermediate_size"])
        if config["hidden_act"] == "gelu":
            self.intermediate_act_fn = nn.GELU()
        else:
            assert False, "hidden_act is set as: {}, please check it..".format(config["hidden_act"])

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LayoutXLMOutput(nn.Layer):
    def __init__(self, config):
        super(LayoutXLMOutput, self).__init__()
        self.dense = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutXLMLayer(nn.Layer):
    def __init__(self, config):
        super(LayoutXLMLayer, self).__init__()
        # since chunk_size_feed_forward is 0 as default, no chunk is needed here.
        self.seq_len_dim = 1
        self.attention = LayoutXLMAttention(config)
        self.add_cross_attention = False  # default as false
        self.intermediate = LayoutXLMIntermediate(config)
        self.output = LayoutXLMOutput(config)

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
        rel_2d_pos=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]
        layer_output = self.feed_forward_chunk(attention_output)

        if output_attentions:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            outputs = [
                layer_output,
            ] + list(outputs)
        else:
            outputs = [layer_output]
        return outputs


class VisualBackbone(nn.Layer):
    def __init__(self, config):
        super(VisualBackbone, self).__init__()
        self.cfg = read_config()
        self.backbone = build_resnet_fpn_backbone(self.cfg)

        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer("pixel_mean", paddle.to_tensor(self.cfg.MODEL.PIXEL_MEAN).reshape([num_channels, 1, 1]))
        self.register_buffer("pixel_std", paddle.to_tensor(self.cfg.MODEL.PIXEL_STD).reshape([num_channels, 1, 1]))
        self.out_feature_key = "p2"
        # is_deterministic is disabled here.
        self.pool = nn.AdaptiveAvgPool2D(config["image_feature_pool_shape"][:2])
        if len(config["image_feature_pool_shape"]) == 2:
            config["image_feature_pool_shape"].append(self.backbone.output_shape()[self.out_feature_key].channels)
        assert self.backbone.output_shape()[self.out_feature_key].channels == config["image_feature_pool_shape"][2]

    def forward(self, images):
        images_input = (paddle.to_tensor(images) - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        features = self.pool(features).flatten(start_axis=2).transpose([0, 2, 1])
        return features


@register_base_model
class LayoutXLMModel(LayoutXLMPretrainedModel):
    """
    The bare LayoutXLM Model outputting raw hidden-states.

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
        with_pool="tanh",
        use_visual_backbone=True,
        **kwargs,
    ):
        super(LayoutXLMModel, self).__init__()
        config = kwargs
        self.config = kwargs
        self.use_visual_backbone = use_visual_backbone
        self.has_visual_segment_embedding = config["has_visual_segment_embedding"]
        self.embeddings = LayoutXLMEmbeddings(config)

        if self.use_visual_backbone is True:
            self.visual = VisualBackbone(config)
            self.visual.stop_gradient = True
            self.visual_proj = nn.Linear(config["image_feature_pool_shape"][-1], config["hidden_size"])

        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = self.create_parameter(
                shape=[
                    config["hidden_size"],
                ],
                dtype=paddle.float32,
            )
        self.visual_LayerNorm = nn.LayerNorm(config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.visual_dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.encoder = LayoutXLMEncoder(config)
        self.pooler = LayoutXLMPooler(config["hidden_size"], with_pool)

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, visual_shape):
        visual_bbox_x = (
            paddle.arange(
                0,
                1000 * (image_feature_pool_shape[1] + 1),
                1000,
                dtype=bbox.dtype,
            )
            // image_feature_pool_shape[1]
        )
        visual_bbox_y = (
            paddle.arange(
                0,
                1000 * (image_feature_pool_shape[0] + 1),
                1000,
                dtype=bbox.dtype,
            )
            // image_feature_pool_shape[0]
        )

        expand_shape = image_feature_pool_shape[0:2]
        visual_bbox = paddle.stack(
            [
                visual_bbox_x[:-1].expand(expand_shape),
                visual_bbox_y[:-1].expand(expand_shape[::-1]).transpose([1, 0]),
                visual_bbox_x[1:].expand(expand_shape),
                visual_bbox_y[1:].expand(expand_shape[::-1]).transpose([1, 0]),
            ],
            axis=-1,
        ).reshape([expand_shape[0] * expand_shape[1], paddle.shape(bbox)[-1]])

        visual_bbox = visual_bbox.expand([visual_shape[0], visual_bbox.shape[0], visual_bbox.shape[1]])
        return visual_bbox

    def _calc_img_embeddings(self, image, bbox, position_ids):
        use_image_info = self.use_visual_backbone and image is not None
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        if use_image_info is True:
            visual_embeddings = self.visual_proj(self.visual(image.astype(paddle.float32)))
            embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        else:
            embeddings = position_embeddings + spatial_position_embeddings

        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        num_position_embeds_diff = new_num_position_embeddings - self.config["max_position_embeddings"]

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config["max_position_embeddings"] = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.weight

        self.embeddings.position_embeddings = nn.Embedding(
            self.config["max_position_embeddings"], self.config["hidden_size"]
        )

        with paddle.no_grad():
            if num_position_embeds_diff > 0:
                self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = old_position_embeddings_weight
            else:
                self.embeddings.position_embeddings.weight = old_position_embeddings_weight[:num_position_embeds_diff]

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        output_hidden_states=False,
        output_attentions=False,
    ):
        input_shape = paddle.shape(input_ids)
        visual_shape = list(input_shape)
        visual_shape[1] = self.config["image_feature_pool_shape"][0] * self.config["image_feature_pool_shape"][1]
        visual_bbox = self._calc_visual_bbox(self.config["image_feature_pool_shape"], bbox, visual_shape)

        final_bbox = paddle.concat([bbox, visual_bbox], axis=1)
        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)

        if self.use_visual_backbone is True:
            visual_attention_mask = paddle.ones(visual_shape)
        else:
            visual_attention_mask = paddle.zeros(visual_shape)

        attention_mask = attention_mask.astype(visual_attention_mask.dtype)

        final_attention_mask = paddle.concat([attention_mask, visual_attention_mask], axis=1)

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand(input_shape)

        visual_position_ids = paddle.arange(0, visual_shape[1]).expand([input_shape[0], visual_shape[1]])
        final_position_ids = paddle.concat([position_ids, visual_position_ids], axis=1)

        if bbox is None:
            bbox = paddle.zeros(input_shape + [4])

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = paddle.concat([text_layout_emb, visual_emb], axis=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config["num_hidden_layers"], -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            head_mask = [None] * self.config["num_hidden_layers"]

        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output, encoder_outputs[1]


class LayoutXLMForTokenClassification(LayoutXLMPretrainedModel):
    def __init__(self, layoutxlm, num_classes=2, dropout=None):
        super(LayoutXLMForTokenClassification, self).__init__()
        self.num_classes = num_classes
        if isinstance(layoutxlm, dict):
            self.layoutxlm = LayoutXLMModel(**layoutxlm)
        else:
            self.layoutxlm = layoutxlm
        self.dropout = nn.Dropout(dropout if dropout is not None else self.layoutxlm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.layoutxlm.config["hidden_size"], num_classes)
        self.classifier.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.layoutxlm.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.layoutxlm.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        outputs = self.layoutxlm(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = input_ids.shape[1]
        # sequence out and image out
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        hidden_states = {
            f"hidden_states_{idx}": outputs[2][f"{idx}_data"]
            for idx in range(self.layoutxlm.config["num_hidden_layers"])
        }
        if self.training:
            outputs = logits, hidden_states
        else:
            outputs = logits

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = (
                    attention_mask.reshape(
                        [
                            -1,
                        ]
                    )
                    == 1
                )
                active_logits = logits.reshape([-1, self.num_classes])[active_loss]
                active_labels = labels.reshape(
                    [
                        -1,
                    ]
                )[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.reshape([-1, self.num_classes]),
                    labels.reshape(
                        [
                            -1,
                        ]
                    ),
                )

            outputs = (loss,) + outputs

        return outputs


class LayoutXLMForSequenceClassification(LayoutXLMPretrainedModel):
    def __init__(self, layoutxlm, num_classes=2, dropout=None):
        super(LayoutXLMForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        if isinstance(layoutxlm, dict):
            self.layoutxlm = LayoutXLMModel(**layoutxlm)
        else:
            self.layoutxlm = layoutxlm
        self.dropout = nn.Dropout(dropout if dropout is not None else self.layoutxlm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.layoutxlm.config["hidden_size"] * 3, num_classes)
        self.classifier.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.layoutxlm.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.layoutxlm.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        input_shape = paddle.shape(input_ids)
        visual_shape = list(input_shape)
        visual_shape[1] = (
            self.layoutxlm.config["image_feature_pool_shape"][0] * self.layoutxlm.config["image_feature_pool_shape"][1]
        )
        visual_bbox = self.layoutxlm._calc_visual_bbox(
            self.layoutxlm.config["image_feature_pool_shape"], bbox, visual_shape
        )

        visual_position_ids = paddle.arange(0, visual_shape[1]).expand([input_shape[0], visual_shape[1]])

        initial_image_embeddings = self.layoutxlm._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )

        outputs = self.layoutxlm(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = input_ids.shape[1]
        # sequence out and image out
        sequence_output, final_image_embeddings = outputs[0][:, :seq_length], outputs[0][:, seq_length:]

        cls_final_output = sequence_output[:, 0, :]

        # average-pool the visual embeddings
        pooled_initial_image_embeddings = initial_image_embeddings.mean(axis=1)
        pooled_final_image_embeddings = final_image_embeddings.mean(axis=1)
        # concatenate with cls_final_output
        sequence_output = paddle.concat(
            [cls_final_output, pooled_initial_image_embeddings, pooled_final_image_embeddings], axis=1
        )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(
                logits.reshape([-1, self.num_classes]),
                labels.reshape(
                    [
                        -1,
                    ]
                ),
            )

            outputs = (loss,) + outputs

        return outputs


class LayoutXLMPredictionHead(Layer):
    """
    Bert Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self, hidden_size, vocab_size, activation, embedding_weights=None):
        super(LayoutXLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = (
            self.create_parameter(shape=[vocab_size, hidden_size], dtype=self.transform.weight.dtype, is_bias=False)
            if embedding_weights is None
            else embedding_weights
        )
        self.decoder_bias = self.create_parameter(shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states, [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states, masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.tensor.matmul(hidden_states, self.decoder_weight, transpose_y=True) + self.decoder_bias
        return hidden_states


class LayoutXLMPretrainingHeads(Layer):
    def __init__(self, hidden_size, vocab_size, activation, embedding_weights=None):
        super(LayoutXLMPretrainingHeads, self).__init__()
        self.predictions = LayoutXLMPredictionHead(hidden_size, vocab_size, activation, embedding_weights)

    def forward(self, sequence_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores


class LayoutXLMForPretraining(LayoutXLMPretrainedModel):
    def __init__(self, layoutxlm):
        super(LayoutXLMForPretraining, self).__init__()
        self.layoutxlm = layoutxlm
        self.cls = LayoutXLMPretrainingHeads(
            self.layoutxlm.config["hidden_size"],
            self.layoutxlm.config["vocab_size"],
            self.layoutxlm.config["hidden_act"],
            embedding_weights=self.layoutxlm.embeddings.word_embeddings.weight,
        )

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.layoutxlm.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_positions=None,
    ):
        outputs = self.layoutxlm(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output, masked_positions)
        return prediction_scores


class BiaffineAttention(nn.Layer):
    """Implements a biaffine attention operator for binary relation classification."""

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = nn.Bilinear(in_features, in_features, out_features, bias_attr=False)
        self.linear = nn.Linear(2 * in_features, out_features)

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(paddle.concat((x_1, x_2), axis=-1))


class REDecoder(nn.Layer):
    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1):
        super(REDecoder, self).__init__()
        self.entity_emb = nn.Embedding(3, hidden_size)
        projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(hidden_size // 2, 2)
        self.loss_fct = CrossEntropyLoss()

    def build_relation(self, relations, entities):
        batch_size, max_seq_len = paddle.shape(entities)[:2]
        new_relations = paddle.full(
            shape=[batch_size, max_seq_len * max_seq_len, 3], fill_value=-1, dtype=relations.dtype
        )
        for b in range(batch_size):
            if entities[b, 0, 0] <= 2:
                entitie_new = paddle.full(shape=[512, 3], fill_value=-1, dtype=entities.dtype)
                entitie_new[0, :] = 2
                entitie_new[1:3, 0] = 0  # start
                entitie_new[1:3, 1] = 1  # end
                entitie_new[1:3, 2] = 0  # label
                entities[b] = entitie_new
            entitie_label = entities[b, 1 : entities[b, 0, 2] + 1, 2]
            all_possible_relations1 = paddle.arange(0, entities[b, 0, 2], dtype=entities.dtype)
            all_possible_relations1 = all_possible_relations1[entitie_label == 1]
            all_possible_relations2 = paddle.arange(0, entities[b, 0, 2], dtype=entities.dtype)
            all_possible_relations2 = all_possible_relations2[entitie_label == 2]

            all_possible_relations = paddle.stack(
                paddle.meshgrid(all_possible_relations1, all_possible_relations2), axis=2
            ).reshape([-1, 2])
            if len(all_possible_relations) == 0:
                all_possible_relations = paddle.full_like(all_possible_relations, fill_value=-1, dtype=entities.dtype)
                all_possible_relations[0, 0] = 0
                all_possible_relations[0, 1] = 1

            relation_head = relations[b, 1 : relations[b, 0, 0] + 1, 0]
            relation_tail = relations[b, 1 : relations[b, 0, 1] + 1, 1]
            positive_relations = paddle.stack([relation_head, relation_tail], axis=1)

            all_possible_relations_repeat = all_possible_relations.unsqueeze(axis=1).tile(
                [1, len(positive_relations), 1]
            )
            positive_relations_repeat = positive_relations.unsqueeze(axis=0).tile([len(all_possible_relations), 1, 1])
            mask = paddle.all(all_possible_relations_repeat == positive_relations_repeat, axis=2)
            negative_mask = paddle.any(mask, axis=1) == False
            negative_relations = all_possible_relations[negative_mask]

            positive_mask = paddle.any(mask, axis=0) == True
            positive_relations = positive_relations[positive_mask]
            if negative_mask.sum() > 0:
                reordered_relations = paddle.concat([positive_relations, negative_relations])
            else:
                reordered_relations = positive_relations

            relation_per_doc_label = paddle.zeros([len(reordered_relations), 1], dtype=reordered_relations.dtype)
            relation_per_doc_label[: len(positive_relations)] = 1
            relation_per_doc = paddle.concat([reordered_relations, relation_per_doc_label], axis=1)
            assert len(relation_per_doc[:, 0]) != 0
            new_relations[b, 0] = paddle.shape(relation_per_doc)[0].astype(new_relations.dtype)
            new_relations[b, 1 : len(relation_per_doc) + 1] = relation_per_doc
            # new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = paddle.full(shape=[7, 2], fill_value=-1, dtype=relations.dtype)
            rel[0, 0] = relations[:, 0][i]
            rel[1, 0] = entities[:, 0][relations[:, 0][i] + 1]
            rel[1, 1] = entities[:, 1][relations[:, 0][i] + 1]
            rel[2, 0] = entities[:, 2][relations[:, 0][i] + 1]
            rel[3, 0] = relations[:, 1][i]
            rel[4, 0] = entities[:, 0][relations[:, 1][i] + 1]
            rel[4, 1] = entities[:, 1][relations[:, 1][i] + 1]
            rel[5, 0] = entities[:, 2][relations[:, 1][i] + 1]
            rel[6, 0] = 1
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        batch_size, max_length, _ = paddle.shape(entities)
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = paddle.full(
            shape=[batch_size, max_length * max_length, 7, 2], fill_value=-1, dtype=entities.dtype
        )
        for b in range(batch_size):
            relation = relations[b, 1 : relations[b, 0, 0] + 1]
            head_entities = relation[:, 0]
            tail_entities = relation[:, 1]
            relation_labels = relation[:, 2]
            entities_start_index = paddle.to_tensor(entities[b, 1 : entities[b, 0, 0] + 1, 0])
            entities_labels = paddle.to_tensor(entities[b, 1 : entities[b, 0, 2] + 1, 2])
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            tmp_hidden_states = hidden_states[b][head_index]
            if len(tmp_hidden_states.shape) == 1:
                tmp_hidden_states = paddle.unsqueeze(tmp_hidden_states, axis=0)
            head_repr = paddle.concat((tmp_hidden_states, head_label_repr), axis=-1)

            tmp_hidden_states = hidden_states[b][tail_index]
            if len(tmp_hidden_states.shape) == 1:
                tmp_hidden_states = paddle.unsqueeze(tmp_hidden_states, axis=0)
            tail_repr = paddle.concat((tmp_hidden_states, tail_label_repr), axis=-1)

            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relation, entities[b])
            if len(pred_relations) > 0:
                pred_relations = paddle.stack(pred_relations)
                all_pred_relations[b, 0, :, :] = paddle.shape(pred_relations)[0].astype(all_pred_relations.dtype)
                all_pred_relations[b, 1 : len(pred_relations) + 1, :, :] = pred_relations
        return loss, all_pred_relations


class LayoutXLMForRelationExtraction(LayoutXLMPretrainedModel):
    def __init__(self, layoutxlm, hidden_size=768, hidden_dropout_prob=0.1, dropout=None):
        super(LayoutXLMForRelationExtraction, self).__init__()
        if isinstance(layoutxlm, dict):
            self.layoutxlm = LayoutXLMModel(**layoutxlm)
        else:
            self.layoutxlm = layoutxlm

        self.extractor = REDecoder(hidden_size, hidden_dropout_prob)

        self.dropout = nn.Dropout(dropout if dropout is not None else self.layoutxlm.config["hidden_dropout_prob"])

    def init_weights(self, layer):
        """Initialize the weights"""
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(paddle.tensor.normal(mean=0.0, std=0.02, shape=layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_value(paddle.tensor.zeros(shape=layer.bias.shape))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(paddle.tensor.normal(mean=0.0, std=0.02, shape=layer.weight.shape))
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(
                    paddle.tensor.normal(mean=0.0, std=0.02, shape=layer.weight[layer._padding_idx].shape)
                )
        elif isinstance(layer, nn.LayerNorm):
            layer.weight.set_value(paddle.tensor.ones(shape=layer.bias.shape))
            layer.bias.set_value(paddle.tensor.zeros(shape=layer.bias.shape))

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.layoutxlm.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids,
        bbox,
        image=None,
        attention_mask=None,
        entities=None,
        relations=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        outputs = self.layoutxlm(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = input_ids.shape[1]
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]

        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.extractor(sequence_output, entities, relations)
        hidden_states = [outputs[2][f"{idx}_data"] for idx in range(self.layoutxlm.config["num_hidden_layers"])]
        hidden_states = paddle.stack(hidden_states, axis=1)

        res = dict(loss=loss, pred_relations=pred_relations, hidden_states=hidden_states)
        return res


class LayoutXLMForQuestionAnswering(LayoutXLMPretrainedModel):
    def __init__(self, layoutxlm, num_classes=2, dropout=None, has_visual_segment_embedding=False):
        super(LayoutXLMForQuestionAnswering, self).__init__()
        self.num_classes = num_classes
        self.layoutxlm = layoutxlm
        self.has_visual_segment_embedding = has_visual_segment_embedding
        self.dropout = nn.Dropout(dropout if dropout is not None else self.layoutxlm.config["hidden_dropout_prob"])
        self.qa_outputs = nn.Linear(self.layoutxlm.config["hidden_size"], num_classes)
        self.qa_outputs.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.layoutxlm.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        start_positions=None,
        end_positions=None,
    ):
        # In LayoutXLM the type vocab size is 1
        token_type_ids = paddle.zeros_like(input_ids)

        outputs = self.layoutxlm(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = input_ids.shape[1]
        # sequence out and image out
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)

        if token_type_ids is not None:
            span_mask = -token_type_ids * 1e8
        else:
            span_mask = 0

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = paddle.split(logits, num_or_sections=2, axis=-1)
        start_logits = start_logits.squeeze(-1) + span_mask
        end_logits = end_logits.squeeze(-1) + span_mask

        outputs = (start_logits, end_logits) + outputs[2:]

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # Sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not total_loss:
            return outputs
        else:
            outputs = (total_loss,) + outputs
            return outputs
