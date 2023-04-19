# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team.
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
from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer

from .. import PretrainedModel, register_base_model
from ..activations import ACT2FN
from .configuration import (
    PROPHETNET_PRETRAINED_INIT_CONFIGURATION,
    PROPHETNET_PRETRAINED_RESOURCE_FILES_MAP,
    ProphetNetConfig,
)

__all__ = [
    "ProphetNetModel",
    "ProphetNetPretrainedModel",
    "ProphetNetEncoder",
    "ProphetNetDecoder",
    "ProphetNetForConditionalGeneration",
]


def ngram_attention_bias(sequence_length, ngram, dtype):
    """
    This function computes the bias for the predict stream
    """
    left_block = paddle.ones((ngram, sequence_length, sequence_length), dtype=dtype) * float("-inf")
    right_block = left_block.detach().clone()
    # create bias
    for stream_idx in range(ngram):
        right_block[stream_idx] = right_block[stream_idx].fill_diagonal_(0, wrap=False)
        left_block[stream_idx] = paddle.triu(left_block[stream_idx], diagonal=-stream_idx + 1)

    left_block[:, :, 0] = 0
    return paddle.concat([left_block, right_block], axis=2)


def compute_relative_buckets(num_buckets, max_distance, relative_positions, is_bidirectional=False):
    """
    This function computes individual parts of the relative position buckets. For more detail, see paper.
    """
    inv_relative_positions = -relative_positions
    rel_positions_bucket = 0

    if is_bidirectional:
        num_buckets = num_buckets // 2
        rel_positions_bucket = (
            rel_positions_bucket
            + paddle.cast(
                paddle.less_than(inv_relative_positions, paddle.zeros_like(inv_relative_positions)), dtype=paddle.int32
            )
            * num_buckets
        )
        inv_relative_positions = paddle.abs(inv_relative_positions)
    else:
        inv_relative_positions = (
            paddle.cast(
                paddle.less_than(paddle.zeros_like(inv_relative_positions), inv_relative_positions), dtype=paddle.int32
            )
            * inv_relative_positions
        )

    max_exact = num_buckets // 2
    is_small = paddle.less_than(inv_relative_positions, paddle.to_tensor(max_exact).cast(dtype=paddle.int32))
    val_if_large = max_exact + paddle.log(
        paddle.cast(inv_relative_positions, dtype=paddle.float32) / max_exact
    ) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    val_if_large_num_buckets = paddle.ones_like(val_if_large) * (num_buckets - 1)
    val_if_large_lt = paddle.cast(paddle.less_than(val_if_large, val_if_large_num_buckets), dtype=paddle.int32)
    val_if_large = (
        paddle.cast(val_if_large_lt * val_if_large, dtype=paddle.int32)
        + (1 - val_if_large_lt) * val_if_large_num_buckets
    )
    rel_positions_bucket = rel_positions_bucket + paddle.where(
        is_small, paddle.cast(inv_relative_positions, dtype=paddle.int32), val_if_large
    )
    return rel_positions_bucket


def compute_all_stream_relative_buckets(num_buckets, max_distance, position_ids):
    """
    This function computes both main and predict relative position buckets. For more detail, see paper.
    """
    # main stream
    main_stream_relative_positions = paddle.tile(
        paddle.unsqueeze(position_ids, axis=1), repeat_times=[1, position_ids.shape[-1], 1]
    )
    main_stream_relative_positions = main_stream_relative_positions - paddle.unsqueeze(position_ids, axis=-1)

    # predicting stream
    predicting_stream_relative_positions = paddle.unsqueeze(
        paddle.concat([position_ids - 1, position_ids], axis=-1), axis=1
    )
    predicting_stream_relative_positions = paddle.tile(
        predicting_stream_relative_positions, repeat_times=[1, position_ids.shape[-1], 1]
    )
    predicting_stream_relative_positions = predicting_stream_relative_positions - paddle.unsqueeze(
        position_ids, axis=-1
    )

    # get both position buckets
    main_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, main_stream_relative_positions, is_bidirectional=False
    )
    predict_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, predicting_stream_relative_positions, is_bidirectional=False
    )
    return main_relative_position_buckets, predict_relative_position_buckets


class ProphetNetPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Prophetnet models. It provides Prophetnet related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    """

    pretrained_init_configuration = PROPHETNET_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = PROPHETNET_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "prophetnet"
    config_class = ProphetNetConfig

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.config.init_std,
                    shape=layer.weight.shape,
                )
            )
            if layer.bias is not None:
                layer.bias.set_value(paddle.tensor.zeros(layer.bias.shape))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.config.decoder_start_token_id has to be defined. "
            "In ProphetNet it is usually set to the pad_token_id. See ProphetNet docs for more information"
        )

        # shift inputs to the right
        shifted_input_ids = paddle.zeros_like(input_ids)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids_mask = paddle.cast(shifted_input_ids == -100, dtype=paddle.int32)
        shifted_input_ids = shifted_input_ids_mask * pad_token_id + (1 - shifted_input_ids_mask) * shifted_input_ids

        assert (
            paddle.sum(paddle.cast(shifted_input_ids >= 0, dtype=paddle.int32)).item() == shifted_input_ids.shape[-1]
        ), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class ProphetNetPositionalEmbeddings(nn.Embedding):
    """
    ProphetNetPositional Embeddings.
    """

    def __init__(self, config: ProphetNetConfig):
        self.max_length = config.max_position_embeddings
        super(ProphetNetPositionalEmbeddings, self).__init__(
            config.max_position_embeddings, config.hidden_size, config.pad_token_id
        )

    def forward(self, inputs_shape, attention_mask=None, past_key_values=None, position_ids=None):
        assert (position_ids is None) or (
            self._padding_idx is None
        ), "If position_ids is pre-computed then padding_idx should not be set."

        if position_ids is None:
            if past_key_values is not None:
                # position_ids is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                prev_num_input_ids = past_key_values[0][0].shape[2]
                num_input_ids = inputs_shape[1] + prev_num_input_ids
                position_ids = paddle.ones((1, 1), dtype="int64") * (int(self._padding_idx + num_input_ids))
            else:
                if attention_mask is None:
                    attention_mask = paddle.ones(inputs_shape, dtype="int64")

                # retrieve position_ids from input_ids / attention_mask
                position_ids = (
                    paddle.cast(
                        paddle.cast(paddle.cumsum(attention_mask, axis=1), dtype=attention_mask.dtype)
                        * attention_mask,
                        dtype=paddle.int64,
                    )
                    + self._padding_idx
                )

                # make sure position_ids are not bigger then max_length
                position_ids = paddle.clip(position_ids, min=0, max=self.max_length - 1)

        return super().forward(position_ids), position_ids

    def _forward(self, position_ids):
        return super().forward(position_ids)


class ProphetNetAttention(Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    """

    def __init__(self, hidden_size, attention_dropout, dropout, num_attn_heads: int):
        super().__init__()
        hidden_size = hidden_size

        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_attn_heads = num_attn_heads
        self.head_dim = hidden_size // num_attn_heads

        assert (
            self.head_dim * num_attn_heads == hidden_size
        ), "`config.hidden_size` must be divisible by `config.num_encoder_attention_heads` and `config.num_decoder_attention_heads`"

        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return paddle.transpose(
            paddle.reshape(tensor, [bsz, seq_len, self.num_attn_heads, self.head_dim]), (0, 2, 1, 3)
        )

    def forward(
        self,
        hidden_states,
        key_value_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        batch_size, tgt_len, hidden_size = hidden_states.shape

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        assert hidden_states.shape == [
            batch_size,
            tgt_len,
            hidden_size,
        ], f"Size of hidden states should be {batch_size, tgt_len, hidden_size}, but is {hidden_states.shape}"

        # previous time steps are cached - no need to recompute key and value if they are static
        query_states = self.query_proj(hidden_states) / (self.head_dim**0.5)

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.key_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.value_proj(key_value_states), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.key_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.value_proj(hidden_states), -1, batch_size)

        if is_cross_attention:
            # if cross_attention save Tuple(paddle.Tensor, paddle.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        # project states into the correct shape
        proj_shape = (batch_size * self.num_attn_heads, -1, self.head_dim)
        query_states = paddle.reshape(self._shape(query_states, tgt_len, batch_size), proj_shape)
        key_states = paddle.reshape(key_states, proj_shape)
        value_states = paddle.reshape(value_states, proj_shape)

        src_len = key_states.shape[1]
        attn_weights = paddle.bmm(query_states, key_states.transpose((0, 2, 1)))
        assert attn_weights.shape == [
            batch_size * self.num_attn_heads,
            tgt_len,
            src_len,
        ], f"`attn_weights` should be of size {batch_size * self.num_attn_heads, tgt_len, src_len}, but is of size {attn_weights.shape}"

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if attention_mask is not None and len(attention_mask.shape) == 0:
            attention_mask = None
        assert attention_mask is None or attention_mask.shape == [
            self.num_attn_heads * batch_size,
            1,
            src_len,
        ], f"`attention_mask` should be `None` or of shape attention_mask.shape == {batch_size * self.num_attn_heads, 1, src_len}, but is {attention_mask.shape}"

        if attention_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1)

        attn_probs = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = paddle.bmm(attn_probs, value_states)
        assert attn_output.shape == [
            batch_size * self.num_attn_heads,
            tgt_len,
            self.head_dim,
        ], f"`attn_output` should be of shape {batch_size * self.num_attn_heads, tgt_len, self.head_dim}, but is of shape {attn_output.shape}"

        attn_output = paddle.reshape(
            paddle.transpose(
                paddle.reshape(attn_output, (batch_size, self.num_attn_heads, tgt_len, self.head_dim)), (0, 2, 1, 3)
            ),
            (batch_size, tgt_len, hidden_size),
        )

        attn_output = self.out_proj(attn_output)

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        return attn_output, past_key_value


class ProphetNetFeedForward(Layer):
    """
    This is the residual two feed-forward layer block based on the original Transformer implementation.
    """

    def __init__(self, hidden_size, activation_function, activation_dropout, dropout, ffn_dim: int):
        super(ProphetNetFeedForward, self).__init__()
        self.activation_fn = ACT2FN[activation_function]
        self.intermediate = nn.Linear(hidden_size, ffn_dim)
        self.output = nn.Linear(ffn_dim, hidden_size)
        self.activation_dropout = activation_dropout
        self.dropout = dropout

    def forward(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.output(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ProphetNetNgramSelfAttention(Layer):
    def __init__(
        self,
        hidden_size,
        num_buckets,
        relative_max_distance,
        num_decoder_attention_heads,
        dropout,
        attention_dropout,
        ngram,
    ):
        super(ProphetNetNgramSelfAttention, self).__init__()

        self.hidden_size = hidden_size

        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.num_attn_heads = num_decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.head_dim = hidden_size // self.num_attn_heads
        self.ngram = ngram

        assert (
            self.head_dim * self.num_attn_heads == hidden_size
        ), "config.hidden_size must be divisible by num_attn_heads"
        # key, value, query projection
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)

        # out projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # rel position embeddings
        self.relative_pos_embeddings = nn.Linear(hidden_size, self.num_buckets * self.num_attn_heads)

    def _shape(self, tensor, seq_len, batch_size):
        return paddle.transpose(
            paddle.reshape(tensor, (batch_size, seq_len, self.num_attn_heads, self.head_dim)), (0, 2, 1, 3)
        )

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[Tensor]] = None,
        attention_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
    ):
        batch_size, ngram_sequence_length, hidden_size = hidden_states.shape

        assert hidden_states.shape == [
            batch_size,
            ngram_sequence_length,
            hidden_size,
        ], f"`hidden_states` should be of shape {batch_size, ngram_sequence_length, hidden_size}, but is of shape {hidden_states.shape}"

        # project
        query_states = self.query_proj(hidden_states)
        key_states = self.key_proj(hidden_states)
        value_states = self.value_proj(hidden_states)

        # normalize
        query_states = query_states / (self.head_dim**0.5)

        # reshape
        query_states = self._shape(query_states, ngram_sequence_length, batch_size)
        key_states = self._shape(key_states, -1, batch_size)
        value_states = self._shape(value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_attn_heads, -1, self.head_dim)

        query_states = paddle.reshape(query_states, proj_shape)
        key_states = paddle.reshape(key_states, proj_shape)
        value_states = paddle.reshape(value_states, proj_shape)

        # chunk into main stream and predict stream
        hidden_states_list = paddle.chunk(hidden_states, 1 + self.ngram, axis=1)

        query_states_list = paddle.chunk(query_states, 1 + self.ngram, axis=1)
        key_states_list = paddle.chunk(key_states, 1 + self.ngram, axis=1)
        value_states_list = paddle.chunk(value_states, 1 + self.ngram, axis=1)

        main_hidden_states, hidden_states_predict_list = hidden_states_list[0], hidden_states_list[1:]
        main_query_states, predict_query_states_list = query_states_list[0], query_states_list[1:]
        main_key_states, predict_key_states_list = key_states_list[0], key_states_list[1:]
        main_value_states, predict_value_states_list = value_states_list[0], value_states_list[1:]

        # saved states are stored with shape (batch_size, num_attn_heads, seq_len, head_dim)
        if past_key_value is not None:
            prev_main_key_states = past_key_value[0].reshape([batch_size * self.num_attn_heads, -1, self.head_dim])
            main_key_states = paddle.concat((prev_main_key_states, main_key_states), axis=1)
            prev_main_value_states = past_key_value[1].reshape([batch_size * self.num_attn_heads, -1, self.head_dim])
            main_value_states = paddle.concat((prev_main_value_states, main_value_states), axis=1)

        # Update cache
        past_key_value = (
            paddle.reshape(main_key_states, (batch_size, self.num_attn_heads, -1, self.head_dim)),
            paddle.reshape(main_value_states, (batch_size, self.num_attn_heads, -1, self.head_dim)),
        )

        # get seq_length of main stream only
        sequence_length = ngram_sequence_length // (1 + self.ngram)

        # MAIN-STREAM
        # main attn weights
        main_attn_weights = paddle.bmm(main_query_states, paddle.transpose(main_key_states, (0, 2, 1)))

        # retrieve relative position embeddings for each layer -> see paper for more details
        main_relative_pos_embeddings = self.get_main_relative_pos_embeddings(
            main_hidden_states, main_attn_weights, position_ids, main_relative_position_buckets
        )

        main_attn_weights = main_attn_weights + main_relative_pos_embeddings

        if attention_mask is not None:
            main_attn_weights = main_attn_weights + attention_mask

        main_attn_probs = F.softmax(main_attn_weights, axis=-1, dtype=main_attn_weights.dtype)

        main_attn_probs = F.dropout(main_attn_probs, p=self.attention_dropout, training=self.training)
        # project to attn_output
        main_attn_output = paddle.bmm(main_attn_probs, main_value_states)

        # reshape so that num_heads dim is merged into last `head_dim` axis
        main_attn_output = paddle.reshape(
            paddle.transpose(
                paddle.reshape(main_attn_output, (batch_size, self.num_attn_heads, sequence_length, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (batch_size, 1, sequence_length, hidden_size),
        )
        main_attn_output = self.out_proj(main_attn_output)

        # PREDICT-STREAM
        # [ngram, B*head, T, c]
        predict_query_states = paddle.reshape(
            paddle.concat(predict_query_states_list, axis=0), (self.ngram, -1, sequence_length, self.head_dim)
        )
        # [ngram, B*head, 2*T, c]
        predict_key_states = paddle.concat(
            [
                paddle.unsqueeze(paddle.concat([main_key_states, key], axis=1), axis=0)
                for key in predict_key_states_list
            ],
            axis=0,
        )

        # [ngram, T, B, C]
        predict_hidden_states = paddle.reshape(
            paddle.concat(hidden_states_predict_list, axis=0), (self.ngram, sequence_length, batch_size, hidden_size)
        )

        # [ngram, B*head, 2*T, c]
        predict_value_states = paddle.concat(
            [
                paddle.unsqueeze(paddle.concat([main_value_states, v_p], axis=1), axis=0)
                for v_p in predict_value_states_list
            ],
            axis=0,
        )

        # [ngram, B*head, T, 2*T]
        predict_attn_weights = paddle.einsum("nbtc,nbsc->nbts", predict_query_states, predict_key_states)

        # [ngram, B*head, T, S]
        # retrieve relative position embeddings for each layer -> see paper for more details
        predict_relative_pos_embeddings = self.get_predict_relative_pos_embeddings(
            predict_hidden_states, predict_attn_weights, position_ids, predict_relative_position_buckets
        )

        # [ngram, B*head, T, 2*T]
        predict_attn_weights = predict_attn_weights + predict_relative_pos_embeddings

        if extended_predict_attention_mask is not None:
            predict_attn_weights = predict_attn_weights + paddle.cast(
                extended_predict_attention_mask, predict_attn_weights.dtype
            )

        predict_attn_probs = F.softmax(predict_attn_weights, axis=-1, dtype=predict_attn_weights.dtype)

        predict_attn_probs = F.dropout(predict_attn_probs, p=self.attention_dropout, training=self.training)
        # project to attention output
        # [ngram, B*head, T, c]
        predict_attn_output = paddle.einsum("nbts,nbsc->nbtc", predict_attn_probs, predict_value_states)

        # reshape so that num_heads dim is merged into last `head_dim` axis
        # [ngram, B, T, C]
        predict_attn_output = paddle.reshape(
            paddle.transpose(
                paddle.reshape(
                    predict_attn_output, (self.ngram, batch_size, self.num_attn_heads, sequence_length, self.head_dim)
                ),
                (1, 0, 3, 2, 4),
            ),
            (batch_size, self.ngram, sequence_length, hidden_size),
        )
        predict_attn_output = self.out_proj(predict_attn_output)

        # concat to single attn output
        # [B, 1+ngram*T, C]
        attn_output = paddle.reshape(
            paddle.concat([main_attn_output, predict_attn_output], axis=1), (batch_size, -1, hidden_size)
        )
        # reshape into better form for `config.output_attentions`
        main_attn_probs = paddle.reshape(main_attn_probs, (batch_size, self.num_attn_heads, sequence_length, -1))
        predict_attn_probs = paddle.transpose(
            paddle.reshape(predict_attn_probs, (self.ngram, batch_size, self.num_attn_heads, sequence_length, -1)),
            (1, 0, 2, 3, 4),
        )

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)

        return attn_output, main_attn_probs, predict_attn_probs, past_key_value

    def get_main_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, main_relative_position_buckets
    ):
        # input hidden_states [B,T,C], input attn_weights [T*head,T,S], input position_ids [B,T] or [1,1]

        if main_relative_position_buckets is None:
            batch_size, sequence_length = hidden_states.shape[:2]
            relative_positions = paddle.tile(
                paddle.unsqueeze(paddle.unsqueeze(paddle.arange(1, attn_weights.shape[-1] + 1), axis=0), axis=0),
                repeat_times=[batch_size, sequence_length, 1],
            )
            relative_positions = relative_positions - paddle.tile(
                paddle.unsqueeze(position_ids, axis=0), repeat_times=[batch_size, sequence_length, 1]
            )  # [B, T, s]
            main_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)  # [B,T,Buckets*head]
        rel_pos_embeddings = paddle.transpose(
            paddle.reshape(
                rel_pos_embeddings, (rel_pos_embeddings.shape[:2] + [self.num_buckets, self.num_attn_heads])
            ),
            (0, 3, 1, 2),
        )  # [B,T,Buckets,head]
        rel_pos_embeddings = rel_pos_embeddings.reshape(attn_weights.shape[:2] + [-1])  # [B*head,T,Buckets]

        main_relative_position_buckets = paddle.cast(
            paddle.reshape(
                paddle.tile(main_relative_position_buckets, repeat_times=[1, self.num_attn_heads, 1]),
                (-1, main_relative_position_buckets.shape[-1]),
            ),
            dtype=paddle.int64,
        )  # [B*head*T, T]
        rel_pos_embeddings = paddle.reshape(
            rel_pos_embeddings, (-1, rel_pos_embeddings.shape[-1])
        )  # [B*head*T,Buckets]

        main_relative_position_buckets_index = paddle.tile(
            main_relative_position_buckets.unsqueeze(2), repeat_times=[1, 1, 2]
        )
        main_relative_position_buckets_index[:, :, 0] = paddle.tile(
            paddle.arange(0, main_relative_position_buckets_index.shape[0]).unsqueeze(1),
            repeat_times=[1, main_relative_position_buckets_index.shape[1]],
        )

        main_relative_pos_embeddings = paddle.reshape(
            paddle.gather_nd(rel_pos_embeddings, index=main_relative_position_buckets_index),
            (attn_weights.shape[:2] + [-1]),
        )
        return main_relative_pos_embeddings

    def get_predict_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, predict_relative_position_buckets
    ):
        # input hidden_states [ngram, T,B,C],
        # input attn_weights [ngram, B*head,T,S],
        # input position_ids [B,T] or [1,1],
        # input predict_relative_position_buckets [B,T, 2*T] or None
        sequence_length, batch_size = hidden_states.shape[1:3]

        if predict_relative_position_buckets is None:
            key_sequence_length = attn_weights.shape[-1]
            assert (
                position_ids[0][0] == key_sequence_length - 1
            ), "`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)"
            relative_positions = paddle.tile(
                paddle.unsqueeze(paddle.unsqueeze(paddle.arange(0, key_sequence_length), axis=0), axis=0),
                repeat_times=[batch_size, sequence_length, 1],
            )

            relative_positions = relative_positions - paddle.tile(
                paddle.unsqueeze(position_ids, axis=0), repeat_times=[batch_size, sequence_length, 1]
            )
            predict_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        hidden_states = paddle.transpose(hidden_states, (0, 2, 1, 3))  # [ngram, B, T, C]
        rel_pos_embeddings = paddle.reshape(
            self.relative_pos_embeddings(hidden_states),
            hidden_states.shape[:-1] + [self.num_buckets, self.num_attn_heads],
        )  # [ngram, B, T, bucket, head]
        rel_pos_embeddings = paddle.reshape(
            paddle.transpose(rel_pos_embeddings, (0, 1, 4, 2, 3)),
            (self.ngram * batch_size * self.num_attn_heads, sequence_length, -1),
        )  # [ngram*B*head, T, bucket]

        predict_relative_position_buckets = paddle.tile(
            paddle.unsqueeze(predict_relative_position_buckets, axis=0),
            repeat_times=[self.ngram, 1, self.num_attn_heads, 1],
        )  # [ngram, B, head*T, S]

        rel_pos_embeddings = paddle.reshape(rel_pos_embeddings, (-1, rel_pos_embeddings.shape[-1]))
        predict_relative_position_buckets = paddle.cast(
            paddle.reshape(predict_relative_position_buckets, (-1, predict_relative_position_buckets.shape[-1])),
            dtype=paddle.int64,
        )  # [ngram*B*head*T, S]

        predict_relative_position_buckets_index = paddle.tile(
            predict_relative_position_buckets.unsqueeze(2), repeat_times=[1, 1, 2]
        )
        predict_relative_position_buckets_index[:, :, 0] = paddle.tile(
            paddle.arange(0, predict_relative_position_buckets_index.shape[0]).unsqueeze(1),
            repeat_times=[1, predict_relative_position_buckets_index.shape[1]],
        )

        predict_relative_pos_embeddings = paddle.reshape(
            paddle.gather_nd(rel_pos_embeddings, index=predict_relative_position_buckets_index),
            (self.ngram, batch_size * self.num_attn_heads, sequence_length, -1),
        )  # [ngram, B*head, T, S]

        return predict_relative_pos_embeddings


class ProphetNetEncoderLayer(Layer):
    """
    Encoder block for Prophetnet
    """

    def __init__(self, config: ProphetNetConfig):
        super(ProphetNetEncoderLayer, self).__init__()
        # 1st residual block
        self.self_attn = ProphetNetAttention(
            config.hidden_size, config.attention_dropout, config.dropout, config.num_encoder_attention_heads
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        # 2nd residual block
        self.feed_forward = ProphetNetFeedForward(
            config.hidden_size,
            config.activation_function,
            config.activation_dropout,
            config.dropout,
            config.encoder_ffn_dim,
        )
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask):
        # 1st residual block
        attention_output, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        hidden_states = self.self_attn_layer_norm(attention_output + hidden_states)

        # 2nd residual block
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)
        return hidden_states


class ProphetNetDecoderLayer(Layer):
    """
    Decoder block for Prophetnet
    """

    def __init__(self, config: ProphetNetConfig):
        super(ProphetNetDecoderLayer, self).__init__()
        # 1st residual block
        self.self_attn = ProphetNetNgramSelfAttention(
            config.hidden_size,
            config.num_buckets,
            config.relative_max_distance,
            config.num_decoder_attention_heads,
            config.dropout,
            config.attention_dropout,
            config.ngram,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        # 2nd residual block
        if config.add_cross_attention:
            self.cross_attn = ProphetNetAttention(
                config.hidden_size, config.attention_dropout, config.dropout, config.num_decoder_attention_heads
            )
            self.cross_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        # 3rd residual block
        self.feed_forward = ProphetNetFeedForward(
            config.hidden_size,
            config.activation_function,
            config.activation_dropout,
            config.dropout,
            config.decoder_ffn_dim,
        )
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attn_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
        past_key_value=None,
        use_cache: bool = True,
    ):
        # 1st residual block
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        ngram_attention_output, self_attn_weights, self_attn_weights_ngram, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            extended_predict_attention_mask=extended_predict_attention_mask,
            main_relative_position_buckets=main_relative_position_buckets,
            predict_relative_position_buckets=predict_relative_position_buckets,
            position_ids=position_ids,
        )
        hidden_states = self.self_attn_layer_norm(hidden_states + ngram_attention_output)

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        if encoder_hidden_states is not None:
            # 2nd residual block
            attention_output, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attn_mask,
                past_key_value=cross_attn_past_key_value,
            )
            hidden_states = self.cross_attn_layer_norm(attention_output + hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # 3rd residual block
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ProphetNetEncoder(ProphetNetPretrainedModel):
    r"""
    word_embeddings  (:obj:`paddle.nn.Embeddings` of shape :obj:`(config.vocab_size, config.hidden_size)`, `optional`):
        The word embedding parameters. This can be used to initialize :class:`~transformers.ProphetNetEncoder` with
        pre-defined word embeddings instead of randomly initialized word embeddings.
    """

    def __init__(self, word_embeddings, config: ProphetNetConfig):
        super(ProphetNetEncoder, self).__init__(config)
        self.init_std = config.init_std
        if word_embeddings is not None:
            self.word_embeddings = word_embeddings
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.position_embeddings = ProphetNetPositionalEmbeddings(config)
        self.embeddings_layer_norm = nn.LayerNorm(config.hidden_size)

        self.layers = nn.LayerList([ProphetNetEncoderLayer(config) for _ in range(config.num_encoder_layers)])

    def forward(self, input_ids=None, attention_mask=None):
        if input_ids is None:
            raise ValueError("Input_ids cannot be None.")
        inputs_embeds = self.word_embeddings(input_ids)

        # prepare attention mask
        if attention_mask is not None:
            extended_attention_mask = (
                paddle.tile(
                    1.0 - attention_mask.unsqueeze(1), repeat_times=[self.config.num_encoder_attention_heads, 1, 1]
                )
            ) * -10000.0
            extended_attention_mask = paddle.cast(extended_attention_mask, dtype=inputs_embeds.dtype)
            extended_attention_mask.stop_gradient = True
        else:
            extended_attention_mask = None

        position_embeddings, position_ids = self.position_embeddings(inputs_embeds.shape[:2])

        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.embeddings_layer_norm(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, attention_mask=extended_attention_mask)
        return hidden_states


class ProphetNetDecoder(ProphetNetPretrainedModel):
    def __init__(self, word_embeddings, config: ProphetNetConfig):
        super(ProphetNetDecoder, self).__init__(config)
        self.init_std = config.init_std
        self.ngram = config.ngram
        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance
        self.dropout = config.dropout
        self.max_target_positions = config.max_position_embeddings
        self.add_cross_attention = config.add_cross_attention
        if word_embeddings is not None:
            self.word_embeddings = word_embeddings
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.position_embeddings = ProphetNetPositionalEmbeddings(config)

        self.ngram_embeddings = nn.Embedding(self.ngram, config.hidden_size)
        self.layers = nn.LayerList([ProphetNetDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.embeddings_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=True,
    ):
        if input_ids is None:
            raise ValueError("Decoder input_ids cannot be None.")
        inputs_embeds = self.word_embeddings(input_ids)
        batch_size, sequence_length = inputs_embeds.shape[:2]

        main_stream_pos_embed, position_ids = self.position_embeddings(
            (batch_size, sequence_length), past_key_values=past_key_values
        )

        if past_key_values is not None:
            main_relative_position_buckets, predict_relative_position_buckets = None, None
        else:
            main_relative_position_buckets, predict_relative_position_buckets = self.compute_buffered_relative_buckets(
                position_ids
            )
        predicting_stream_pos_embed = self.position_embeddings._forward(position_ids + 1)

        # add position embeddings
        hidden_states = inputs_embeds + main_stream_pos_embed

        ngram_embeddings = self.ngram_embeddings.weight

        # prepare attention mask
        if past_key_values is not None:
            assert (
                hidden_states.shape[1] == 1
            ), "At the moment `use_cache` is only supported for `decoder_input_ids` of length 1"

            ngram_hidden_states = [
                paddle.tile(
                    (ngram_embeddings[ngram - 1] + predicting_stream_pos_embed), repeat_times=[batch_size, 1, 1]
                )
                for ngram in range(self.ngram)
            ]
            extended_attention_mask = None
            extended_predict_attention_mask = None
        else:
            ngram_hidden_states = [
                (ngram_embeddings[ngram - 1] + predicting_stream_pos_embed) for ngram in range(self.ngram)
            ]
            extended_attention_mask = self.prepare_attention_mask(hidden_states, attention_mask)
            extended_predict_attention_mask = self.prepare_predict_attention_mask(hidden_states, attention_mask)
            extended_attention_mask.stop_gradient = True
            extended_predict_attention_mask.stop_gradient = True

        # prepare encoder attention mask
        if encoder_attention_mask is not None:
            extended_encoder_attention_mask = (
                1.0
                - paddle.tile(
                    encoder_attention_mask[:, None, :], repeat_times=[self.config.num_decoder_attention_heads, 1, 1]
                )
            ) * -10000.0
            extended_encoder_attention_mask = paddle.cast(extended_encoder_attention_mask, dtype=inputs_embeds.dtype)
        else:
            extended_encoder_attention_mask = None

        hidden_states = paddle.concat([hidden_states] + ngram_hidden_states, axis=1)

        if self.embeddings_layer_norm:
            hidden_states = self.embeddings_layer_norm(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        present_key_values = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attn_mask=extended_encoder_attention_mask,
                extended_predict_attention_mask=extended_predict_attention_mask,
                main_relative_position_buckets=main_relative_position_buckets,
                predict_relative_position_buckets=predict_relative_position_buckets,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                present_key_values += (layer_outputs[1],)

        last_hidden_state = hidden_states[:, :sequence_length]  # 1-gram
        last_hidden_state_ngram = hidden_states[:, sequence_length:] if self.ngram > 0 else None  # 2-gram
        return tuple(v for v in [last_hidden_state, last_hidden_state_ngram, present_key_values] if v is not None)

    def compute_buffered_relative_buckets(self, position_ids):
        batch_size, sequence_length = position_ids.shape

        if not hasattr(self, "_main_relative_buckets") or self._main_relative_buckets is None:
            position_ids = paddle.tile(paddle.arange(1, self.max_target_positions + 1), repeat_times=[1, 1])
            self._main_relative_buckets, self._predict_relative_buckets = compute_all_stream_relative_buckets(
                self.num_buckets, self.relative_max_distance, position_ids
            )

        # buffer relative buckets
        main_relative_buckets = paddle.tile(
            self._main_relative_buckets[:, :sequence_length, :sequence_length], repeat_times=[batch_size, 1, 1]
        )
        predict_relative_buckets = paddle.tile(
            paddle.concat(
                [
                    self._predict_relative_buckets[:, :sequence_length, :sequence_length],
                    self._predict_relative_buckets[
                        :, :sequence_length, self.max_target_positions : self.max_target_positions + sequence_length
                    ],
                ],
                axis=2,
            ),
            repeat_times=[batch_size, 1, 1],
        )

        return main_relative_buckets, predict_relative_buckets

    def prepare_attention_mask(self, hidden_states, attention_mask):
        batch_size, seq_length = hidden_states.shape[:2]

        # get causal mask
        if not hasattr(self, "_causal_mask") or self._causal_mask is None:
            causal_mask = paddle.full(
                (self.max_target_positions, self.max_target_positions), -float("inf"), dtype=hidden_states.dtype
            )
            self._causal_mask = paddle.triu(causal_mask, 1)
        extended_causal_mask = paddle.expand(
            self._causal_mask[:seq_length, :seq_length].unsqueeze(0), shape=[batch_size, seq_length, seq_length]
        )

        # add usual attention mask
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask.unsqueeze(1)) * -10000.0
            extended_attention_mask = extended_causal_mask + extended_attention_mask
        else:
            extended_attention_mask = extended_causal_mask
        return paddle.cast(
            paddle.tile(extended_attention_mask, repeat_times=[self.config.num_decoder_attention_heads, 1, 1]),
            dtype=hidden_states.dtype,
        )

    def prepare_predict_attention_mask(self, hidden_states, attention_mask):
        batch_size, seq_length = hidden_states.shape[:2]

        # get causal mask
        if not hasattr(self, "_predict_causal_mask") or self._predict_causal_mask is None:
            self._predict_causal_mask = ngram_attention_bias(
                self.max_target_positions, self.ngram, hidden_states.dtype
            )
        predict_causal_mask = paddle.concat(
            [
                self._predict_causal_mask[:, :seq_length, :seq_length],
                self._predict_causal_mask[
                    :, :seq_length, self.max_target_positions : self.max_target_positions + seq_length
                ],
            ],
            axis=-1,
        )
        extended_predict_causal_mask = paddle.expand(
            predict_causal_mask[:, None, :, :],
            shape=predict_causal_mask.shape[:1] + [batch_size] + predict_causal_mask.shape[1:],
        )

        # add usual attention mask
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[None, :, None, :]) * -10000.0
            extended_attention_mask = extended_attention_mask.expand((self.ngram, batch_size, seq_length, seq_length))
            # predicted stream attention_mask should always be 0
            extended_attention_mask = paddle.concat(
                [extended_attention_mask, paddle.zeros_like(extended_attention_mask)], axis=-1
            )
            extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
        else:
            extended_predict_attention_mask = extended_predict_causal_mask
        return paddle.cast(
            extended_predict_attention_mask.tile([1, self.config.num_decoder_attention_heads, 1, 1]),
            dtype=hidden_states.dtype,
        )


@register_base_model
class ProphetNetModel(ProphetNetPretrainedModel):
    def __init__(self, config: ProphetNetConfig):
        super(ProphetNetModel, self).__init__(config)
        self.init_std = config.init_std
        self.eps = config.eps
        self.pad_token_id = config.pad_token_id
        self.disable_ngram_loss = config.disable_ngram_loss
        self.decoder_start_token_id = config.decoder_start_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.encoder = ProphetNetEncoder(self.word_embeddings, config)

        self.decoder = ProphetNetDecoder(self.word_embeddings, config)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_output: Optional[Tuple] = None,
        use_cache=True,
        past_key_values=None,
    ):
        if attention_mask is None:
            assert input_ids is not None, "input_ids should be " "specified when generating attention_mask"
            attention_mask = paddle.cast(input_ids != self.pad_token_id, dtype=paddle.get_default_dtype())

        if decoder_attention_mask is None:
            assert decoder_input_ids is not None, (
                "decoder_input_ids should be " "specified when generating decoder_attention_mask"
            )
            decoder_attention_mask = paddle.cast(
                decoder_input_ids != self.pad_token_id, dtype=paddle.get_default_dtype()
            )
        if encoder_output is None:
            encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        return decoder_outputs + (encoder_output,)


class Linear_wo_bias(Layer):
    def __init__(self, in_features, out_features, weight_attr=None, name=None):
        super(Linear_wo_bias, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self.weight = self.create_parameter(
            shape=[in_features, out_features], attr=self._weight_attr, dtype=self._dtype, is_bias=False
        )
        self.name = name

    def forward(self, input):
        out = F.linear(x=input, weight=self.weight, name=self.name)
        return out

    def extra_repr(self):
        name_str = ", name={}".format(self.name) if self.name else ""
        return "in_features={}, out_features={}, dtype={}{}".format(
            self.weight.shape[0], self.weight.shape[1], self._dtype, name_str
        )


class ProphetNetForConditionalGeneration(ProphetNetPretrainedModel):
    def __init__(self, config: ProphetNetConfig):
        super(ProphetNetForConditionalGeneration, self).__init__(config)
        self.prophetnet = ProphetNetModel(config)
        self.padding_idx = self.prophetnet.word_embeddings._padding_idx

        self.lm_head = Linear_wo_bias(config.hidden_size, config.vocab_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_output=None,
        labels=None,
        use_cache=True,
        past_key_values=None,
    ):
        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        outputs = self.prophetnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_output=encoder_output,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        batch_size, sequence_length = decoder_input_ids.shape

        predicting_streams = paddle.reshape(outputs[1], (batch_size, self.config.ngram, sequence_length, -1))
        predict_logits = self.lm_head(predicting_streams)

        logits = predict_logits[:, 0]
        if use_cache:
            past_key_values = outputs[2]
            return logits, past_key_values, predict_logits
        else:
            return logits, predict_logits

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        cache=None,
        use_cache=None,
        encoder_output=None,
    ):
        assert encoder_output is not None, "`encoder_output` have to be passed for generation."
        if cache is not None:
            decoder_input_ids = decoder_input_ids[:, -1].unsqueeze(-1)

        # first step, decoder_cached_states are empty
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "decoder_input_ids": decoder_input_ids,
            "encoder_output": encoder_output,
            "decoder_attention_mask": decoder_attention_mask,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "past_key_values": cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels):
        return self._shift_right(labels)

    def get_encoder(self):
        return self.prophetnet.encoder

    def get_decoder(self):
        return self.prophetnet.decoder

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(getattr(self, self.base_model_prefix), name)
