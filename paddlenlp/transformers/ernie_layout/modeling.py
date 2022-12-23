# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
""" Modeling classes for ErnieLayout model."""

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer

from paddlenlp.utils.log import logger

from .. import PretrainedModel, register_base_model
from .configuration import (
    ERNIE_LAYOUT_PRETRAINED_INIT_CONFIGURATION,
    ERNIE_LAYOUT_PRETRAINED_RESOURCE_FILES_MAP,
    ErnieLayoutConfig,
)
from .visual_backbone import ResNet

__all__ = [
    "ErnieLayoutModel",
    "ErnieLayoutPretrainedModel",
    "ErnieLayoutForTokenClassification",
    "ErnieLayoutForSequenceClassification",
    "ErnieLayoutForPretraining",
    "ErnieLayoutForQuestionAnswering",
    "UIEX",
]


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for small
    absolute relative_position and larger buckets for larger absolute relative_positions. All relative positions
    >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket. This should
    allow for more graceful generalization to longer sequences than the model has been trained on.

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """

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


class ErnieLayoutPooler(Layer):
    def __init__(self, hidden_size, with_pool):
        super(ErnieLayoutPooler, self).__init__()
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


class ErnieLayoutEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", paddle.arange(config.max_position_embeddings).expand((1, -1)))

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
        return (
            left_position_embeddings,
            upper_position_embeddings,
            right_position_embeddings,
            lower_position_embeddings,
            h_position_embeddings,
            w_position_embeddings,
        )

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

        x1, y1, x2, y2, h, w = self.embeddings._cal_spatial_position_embeddings(bbox)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + x1 + y1 + x2 + y2 + h + w + token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ErnieLayoutPretrainedModel(PretrainedModel):
    model_config_file = "config.json"
    pretrained_init_configuration = ERNIE_LAYOUT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ERNIE_LAYOUT_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "ernie_layout"
    config_class = ErnieLayoutConfig

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


class ErnieLayoutSelfOutput(nn.Layer):
    def __init__(self, config):
        super(ErnieLayoutSelfOutput, self).__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ErnieLayoutSelfAttention(nn.Layer):
    def __init__(self, config):
        super(ErnieLayoutSelfAttention, self).__init__()
        if config["hidden_size"] % config["num_attention_heads"] != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size {} is not a multiple of the number of attention "
                "heads {}".format(config["hidden_size"], config["num_attention_heads"])
            )
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = int(config["hidden_size"] / config["num_attention_heads"])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config["has_relative_attention_bias"]
        self.has_spatial_attention_bias = config["has_spatial_attention_bias"]

        self.query = nn.Linear(config["hidden_size"], self.all_head_size)
        self.key = nn.Linear(config["hidden_size"], self.all_head_size)
        self.value = nn.Linear(config["hidden_size"], self.all_head_size)

        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])

    def transpose_for_scores(self, x):
        x = x.reshape([paddle.shape(x)[0], paddle.shape(x)[1], self.num_attention_heads, self.attention_head_size])
        return x.transpose([0, 2, 1, 3])

    def compute_qkv(self, hidden_states):
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
        past_key_values=None,
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
        attention_scores = paddle.matmul(query_layer, key_layer, transpose_y=True)

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
        context_layer = context_layer.reshape(
            [paddle.shape(context_layer)[0], paddle.shape(context_layer)[1], self.all_head_size]
        )

        if output_attentions:
            outputs = [context_layer, attention_probs]
        else:
            outputs = [context_layer]
        return outputs


class ErnieLayoutAttention(nn.Layer):
    def __init__(self, config):
        super(ErnieLayoutAttention, self).__init__()
        self.self = ErnieLayoutSelfAttention(config)
        self.output = ErnieLayoutSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
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
            past_key_values,
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


class ErnieLayoutEncoder(nn.Layer):
    def __init__(self, config):
        super(ErnieLayoutEncoder, self).__init__()
        self.config = config
        self.layer = nn.LayerList([ErnieLayoutLayer(config) for _ in range(config["num_hidden_layers"])])

        self.has_relative_attention_bias = config["has_relative_attention_bias"]
        self.has_spatial_attention_bias = config["has_spatial_attention_bias"]
        if self.has_relative_attention_bias:
            self.rel_pos_bins = config["rel_pos_bins"]
            self.max_rel_pos = config["max_rel_pos"]
            self.rel_pos_onehot_size = config["rel_pos_bins"]
            self.rel_pos_bias = paddle.create_parameter(
                shape=[self.rel_pos_onehot_size, config["num_attention_heads"]], dtype=paddle.get_default_dtype()
            )

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config["max_rel_2d_pos"]
            self.rel_2d_pos_bins = config["rel_2d_pos_bins"]
            self.rel_2d_pos_onehot_size = config["rel_2d_pos_bins"]
            self.rel_pos_x_bias = paddle.create_parameter(
                shape=[self.rel_2d_pos_onehot_size, config["num_attention_heads"]], dtype=paddle.get_default_dtype()
            )
            self.rel_pos_y_bias = paddle.create_parameter(
                shape=[self.rel_2d_pos_onehot_size, config["num_attention_heads"]], dtype=paddle.get_default_dtype()
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
        rel_pos = paddle.matmul(rel_pos, self.rel_pos_bias).transpose([0, 3, 1, 2])
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
        rel_pos_x = paddle.matmul(rel_pos_x, self.rel_pos_x_bias).transpose([0, 3, 1, 2])
        rel_pos_y = paddle.matmul(rel_pos_y, self.rel_pos_y_bias).transpose([0, 3, 1, 2])
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
            past_key_values = past_key_values[i] if past_key_values is not None else None

            # gradient_checkpointing is set as False here so we remove some codes here
            hidden_save["input_attention_mask"] = attention_mask
            hidden_save["input_layer_head_mask"] = layer_head_mask
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values,
                output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
            )

            hidden_states = layer_outputs[0]

            hidden_save["{}_data".format(i)] = hidden_states

        return (hidden_states,)


class ErnieLayoutIntermediate(nn.Layer):
    def __init__(self, config):
        super(ErnieLayoutIntermediate, self).__init__()
        self.dense = nn.Linear(config["hidden_size"], config["intermediate_size"])
        if config["hidden_act"] == "gelu":
            self.intermediate_act_fn = nn.GELU()
        else:
            assert False, "hidden_act is set as: {}, please check it..".format(config["hidden_act"])

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ErnieLayoutOutput(nn.Layer):
    def __init__(self, config):
        super(ErnieLayoutOutput, self).__init__()
        self.dense = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ErnieLayoutLayer(nn.Layer):
    def __init__(self, config):
        super(ErnieLayoutLayer, self).__init__()
        # since chunk_size_feed_forward is 0 as default, no chunk is needed here.
        self.seq_len_dim = 1
        self.attention = ErnieLayoutAttention(config)
        self.add_cross_attention = False  # default as false
        self.intermediate = ErnieLayoutIntermediate(config)
        self.output = ErnieLayoutOutput(config)

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
        past_key_values=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_values = past_key_values[:2] if past_key_values is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_values=self_attn_past_key_values,
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

        self.backbone = ResNet(layers=101)

        self.register_buffer("pixel_mean", paddle.to_tensor([103.53, 116.28, 123.675]).reshape([3, 1, 1]))
        self.register_buffer("pixel_std", paddle.to_tensor([57.375, 57.12, 58.395]).reshape([3, 1, 1]))

        self.pool = nn.AdaptiveAvgPool2D(config["image_feature_pool_shape"][:2])

    def forward(self, images):
        images_input = (paddle.to_tensor(images) - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = self.pool(features).flatten(start_axis=2).transpose([0, 2, 1])
        return features


@register_base_model
class ErnieLayoutModel(ErnieLayoutPretrainedModel):
    """
    The bare ErnieLayout Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
       vocab_size (`int`, *optional*, defaults to 250002):
            Vocabulary size of the ErnieLayout model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ErnieLayoutModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 514):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 514 or 1028 or 2056).
        type_vocab_size (`int`, *optional*, defaults to 100):
            The vocabulary size of the `token_type_ids` passed when calling [`ErnieModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
    """

    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutModel, self).__init__(config)
        self.has_visual_segment_embedding = config["has_visual_segment_embedding"]
        self.embeddings = ErnieLayoutEmbeddings(config)

        self.visual = VisualBackbone(config)
        self.visual_proj = nn.Linear(config["image_feature_pool_shape"][-1], config["hidden_size"])
        self.visual_act_fn = nn.GELU()
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = self.create_parameter(
                shape=[
                    config["hidden_size"],
                ],
                dtype=self.embedding.weight.dtype,
            )
        self.visual_LayerNorm = nn.LayerNorm(config["hidden_size"], epsilon=config["layer_norm_eps"])
        self.visual_dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.encoder = ErnieLayoutEncoder(config)
        self.pooler = ErnieLayoutPooler(config["hidden_size"], "tanh")

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        x1, y1, x2, y2, h, w = self.embeddings._cal_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + x1 + y1 + x2 + y2 + w + h + token_type_embeddings

        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        if image is not None:
            visual_embeddings = self.visual_act_fn(self.visual_proj(self.visual(image.astype(paddle.float32))))
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        x1, y1, x2, y2, h, w = self.embeddings._cal_spatial_position_embeddings(bbox)
        if image is not None:
            embeddings = visual_embeddings + position_embeddings + x1 + y1 + x2 + y2 + w + h
        else:
            embeddings = position_embeddings + x1 + y1 + x2 + y2 + w + h

        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
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

        visual_attention_mask = paddle.ones(visual_shape)

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
        return sequence_output, pooled_output


class ErnieLayoutForSequenceClassification(ErnieLayoutPretrainedModel):
    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutForSequenceClassification, self).__init__(config)
        self.ernie_layout = ErnieLayoutModel(config)
        self.num_labels = config.num_labels
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config["hidden_size"] * 3, config.num_labels)

    def get_input_embeddings(self):
        return self.ernie_layout.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.ernie_layout.resize_position_embeddings(new_num_position_embeddings)

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
            self.ernie_layout.config["image_feature_pool_shape"][0]
            * self.ernie_layout.config["image_feature_pool_shape"][1]
        )
        visual_bbox = self.ernie_layout._calc_visual_bbox(
            self.ernie_layout.config["image_feature_pool_shape"], bbox, visual_shape
        )

        visual_position_ids = paddle.arange(0, visual_shape[1]).expand([input_shape[0], visual_shape[1]])

        initial_image_embeddings = self.ernie_layout._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )

        outputs = self.ernie_layout(
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
                logits.reshape([-1, self.num_labels]),
                labels.reshape(
                    [
                        -1,
                    ]
                ),
            )

            outputs = (loss,) + outputs

        return outputs


class ErnieLayoutPredictionHead(Layer):
    """
    Bert Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self, hidden_size, vocab_size, activation, embedding_weights=None):
        super(ErnieLayoutPredictionHead, self).__init__()
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


class ErnieLayoutPretrainingHeads(Layer):
    def __init__(self, hidden_size, vocab_size, activation, embedding_weights=None):
        super(ErnieLayoutPretrainingHeads, self).__init__()
        self.predictions = ErnieLayoutPredictionHead(hidden_size, vocab_size, activation, embedding_weights)

    def forward(self, sequence_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores


class ErnieLayoutForPretraining(ErnieLayoutPretrainedModel):
    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutForPretraining, self).__init__(config)
        self.ernie_layout = ErnieLayoutModel(config)
        self.cls = ErnieLayoutPretrainingHeads(
            config.hidden_size,
            config.vocab_size,
            config.hidden_act,
            embedding_weights=self.ernie_layout.embeddings.word_embeddings.weight,
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
        self.ernie_layout.resize_position_embeddings(new_num_position_embeddings)

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
        outputs = self.ernie_layout(
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


class ErnieLayoutForTokenClassification(ErnieLayoutPretrainedModel):
    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.ernie_layout = ErnieLayoutModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config["hidden_size"], config.num_labels)
        self.classifier.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.ernie_layout.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.ernie_layout.resize_position_embeddings(new_num_position_embeddings)

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
        outputs = self.ernie_layout(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = paddle.shape(input_ids)[1]
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)

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
                active_logits = logits.reshape([-1, self.num_labels])[active_loss]
                active_labels = labels.reshape(
                    [
                        -1,
                    ]
                )[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.reshape([-1, self.num_labels]),
                    labels.reshape(
                        [
                            -1,
                        ]
                    ),
                )

            outputs = (loss,) + outputs

        return outputs


class ErnieLayoutForQuestionAnswering(ErnieLayoutPretrainedModel):
    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels
        self.ernie_layout = ErnieLayoutModel(config)
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.qa_outputs = nn.Linear(config["hidden_size"], 2)
        self.qa_outputs.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.ernie_layout.embeddings.word_embeddings

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
        outputs = self.ernie_layout(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        seq_length = paddle.shape(input_ids)[1]
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
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
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


class UIEX(ErnieLayoutPretrainedModel):
    def __init__(self, config: ErnieLayoutConfig):
        super(UIEX, self).__init__(config)
        self.ernie_layout = ErnieLayoutModel(config)
        self.linear_start = nn.Linear(config.hidden_size, 1)
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, bbox=None, image=None):
        sequence_output, _ = self.ernie_layout(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            image=image,
        )
        seq_length = paddle.shape(input_ids)[1]
        sequence_output = sequence_output[:, :seq_length]
        start_logits = self.linear_start(sequence_output)
        start_logits = paddle.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = paddle.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob
