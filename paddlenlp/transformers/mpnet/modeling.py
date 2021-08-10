# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

import copy
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .. import PretrainedModel, register_base_model

from ..nezha.modeling import ACT2FN

__all__ = [
    "MPNetModel",
    "MPNetPretrainedModel",
    "MPNetForMaskedLM",
    "MPNetForSequenceClassification",
    "MPNetForMultipleChoice",
    "MPNetForTokenClassification",
    "MPNetForQuestionAnswering",
]


def create_position_ids_from_input_ids(input_ids, padding_idx=1):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`. :param paddle.Tensor x: :return paddle.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = (input_ids != padding_idx).astype(paddle.int64)
    incremental_indices = paddle.cumsum(mask, axis=1).astype(mask.dtype) * mask
    return incremental_indices.astype(paddle.int64) + padding_idx


class MPNetEmbeddings(nn.Layer):
    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            hidden_dropout_prob=0.1,
            max_position_embeddings=514,
            layer_norm_eps=1e-5,
            pad_token_id=1, ):
        super(MPNetEmbeddings, self).__init__()
        self.padding_idx = pad_token_id
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size, padding_idx=self.padding_idx)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):

        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids,
                                                              self.padding_idx)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MPNetAttention(nn.Layer):
    def __init__(
            self,
            hidden_size=768,
            num_attention_heads=12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-5, ):
        super(MPNetAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scale = self.attention_head_size**-0.5
        self.q = nn.Linear(hidden_size, self.all_head_size)
        self.k = nn.Linear(hidden_size, self.all_head_size)
        self.v = nn.Linear(hidden_size, self.all_head_size)
        self.o = nn.Linear(hidden_size, hidden_size)

        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = x.reshape(new_x_shape)
        return x.transpose(perm=(0, 2, 1, 3))

    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attention_scores = paddle.matmul(q, k, transpose_y=True) * self.scale

        if position_bias is not None:
            attention_scores += position_bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, axis=-1)

        attention_probs = self.attention_dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, v)

        context_layer = context_layer.transpose(perm=(0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        projected_context_layer = self.o(context_layer)
        projected_context_layer_dropout = self.output_dropout(
            projected_context_layer)
        layer_normed_context_layer = self.layer_norm(
            hidden_states + projected_context_layer_dropout)

        return layer_normed_context_layer, attention_scores


class MPNetLayer(nn.Layer):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps, ):
        super(MPNetLayer, self).__init__()
        self.attention = MPNetAttention(
            hidden_size,
            num_attention_heads,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps, )
        self.ffn = nn.Linear(hidden_size, intermediate_size)
        self.ffn_output = nn.Linear(intermediate_size, hidden_size)
        self.activation = ACT2FN[hidden_act]
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        attention_output, layer_att = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias)

        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)

        ffn_output_dropout = self.dropout(ffn_output)
        hidden_states = self.layer_norm(ffn_output_dropout + attention_output)

        return hidden_states, layer_att


class MPNetEncoder(nn.Layer):
    def __init__(
            self,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            relative_attention_num_buckets,
            layer_norm_eps, ):
        super(MPNetEncoder, self).__init__()
        layer = MPNetLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            layer_norm_eps, )
        self.layer = nn.LayerList(
            [copy.deepcopy(layer) for _ in range(num_hidden_layers)])
        self.relative_attention_bias = nn.Embedding(
            relative_attention_num_buckets, num_attention_heads)

    def forward(self, hidden_states, attention_mask=None):
        position_bias = self.compute_position_bias(hidden_states)
        all_encoder_layers = []
        all_encoder_att = []
        for i, layer_module in enumerate(self.layer):
            all_encoder_layers.append(hidden_states)
            hidden_states, layer_att = layer_module(
                all_encoder_layers[i], attention_mask, position_bias)
            all_encoder_att.append(layer_att)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_att

    def compute_position_bias(self, x, position_ids=None, num_buckets=32):
        bsz, qlen, klen = x.shape[0], x.shape[1], x.shape[1]
        if position_ids is not None:
            context_position = position_ids.unsqueeze(2)
            memory_position = position_ids.unsqueeze(1)
        else:
            context_position = paddle.arange(qlen).unsqueeze(1)
            memory_position = paddle.arange(klen).unsqueeze(0)

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(
            relative_position, num_buckets=num_buckets)

        values = self.relative_attention_bias(rp_bucket)
        values = values.transpose(perm=[2, 0, 1]).unsqueeze(0)
        values = values.expand(shape=(bsz, values.shape[1], qlen, klen))
        return values

    @staticmethod
    def relative_position_bucket(relative_position,
                                 num_buckets=32,
                                 max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).astype(paddle.int64) * num_buckets
        n = paddle.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            paddle.log(n.astype(paddle.float32) / max_exact) / math.log(
                max_distance / max_exact) *
            (num_buckets - max_exact)).astype(paddle.int64)

        val_if_large = paddle.minimum(
            val_if_large, paddle.full_like(val_if_large, num_buckets - 1))
        ret += paddle.where(is_small, n, val_if_large)
        return ret


class MPNetPooler(nn.Layer):
    def __init__(self, hidden_size):
        super(MPNetPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MPNetPretrainedModel(PretrainedModel):
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "mpnet-base": {
            "vocab_size": 30527,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 514,
            "relative_attention_num_buckets": 32,
            "layer_norm_eps": 1e-05,
            "initializer_range": 0.02,
            "pad_token_id": 1,
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "mpnet-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/mpnet/mpnet-base/model_state.pdparams",
        }
    }
    base_model_prefix = "mpnet"

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.mpnet.config["initializer_range"],
                        shape=layer.weight.shape, ))


@register_base_model
class MPNetModel(MPNetPretrainedModel):
    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=514,
            initializer_range=0.02,
            relative_attention_num_buckets=32,
            layer_norm_eps=1e-5,
            pad_token_id=1, ):
        super(MPNetModel, self).__init__()
        self.initializer_range = initializer_range
        self.embeddings = MPNetEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            layer_norm_eps,
            pad_token_id, )
        self.encoder = MPNetEncoder(
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            relative_attention_num_buckets,
            layer_norm_eps, )

        self.pooler = MPNetPooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.embeddings.padding_idx
                              ).astype(input_ids.dtype)

        if attention_mask.ndim == 2:
            extended_attention_mask = attention_mask.unsqueeze(axis=[1, 2])
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, position_ids)

        encoder_outputs, _ = self.encoder(embedding_output,
                                          extended_attention_mask)

        sequence_output = encoder_outputs[-1]
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class MPNetLMHead(nn.Layer):
    def __init__(
            self,
            hidden_size,
            vocab_size,
            hidden_act="gelu",
            embedding_weights=None,
            layer_norm_eps=1e-5, ):
        super(MPNetLMHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = ACT2FN[hidden_act]
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

        self.decoder_weight = embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        hidden_states = (paddle.matmul(
            hidden_states, self.decoder_weight, transpose_y=True) +
                         self.decoder_bias)

        return hidden_states


class MPNetForMaskedLM(MPNetPretrainedModel):
    def __init__(self, mpnet):
        super(MPNetForMaskedLM, self).__init__()
        self.mpnet = mpnet
        self.lm_head = MPNetLMHead(
            self.mpnet.config["hidden_size"],
            self.mpnet.config["vocab_size"],
            self.mpnet.config["hidden_act"],
            self.mpnet.embeddings.word_embeddings.weight,
            self.mpnet.config["layer_norm_eps"], )

        self.apply(self.init_weights)

    def forward(
            self,
            input_ids,
            position_ids=None,
            attention_mask=None,
            labels=None, ):
        sequence_output, pooled_output = self.mpnet(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.reshape(shape=(
                    -1, self.mpnet.config["vocab_size"])),
                labels.reshape(shape=(-1, )), )
            return masked_lm_loss, prediction_scores, sequence_output

        return prediction_scores, sequence_output


class MPNetForSequenceClassification(MPNetPretrainedModel):
    def __init__(self, mpnet, num_classes=2, dropout=None):
        super(MPNetForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.mpnet = mpnet
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.mpnet.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.mpnet.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        _, pooled_output = self.mpnet(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits


class MPNetForMultipleChoice(MPNetPretrainedModel):
    def __init__(self, mpnet, num_choices=2, dropout=None):
        super(MPNetForMultipleChoice, self).__init__()
        self.num_choices = num_choices
        self.mpnet = mpnet
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.mpnet.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.mpnet.config["hidden_size"], 1)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        # input_ids: [bs, num_choice, seq_l]
        input_ids = input_ids.reshape(shape=(
            -1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if position_ids is not None:
            position_ids = position_ids.reshape(shape=(-1,
                                                       position_ids.shape[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(
                shape=(-1, attention_mask.shape[-1]))

        _, pooled_output = self.mpnet(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(
            shape=(-1, self.num_choices))  # logits: (bs, num_choice)

        return reshaped_logits


class MPNetForTokenClassification(MPNetPretrainedModel):
    def __init__(self, mpnet, num_classes, dropout=None):
        super(MPNetForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.mpnet = mpnet
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.mpnet.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.mpnet.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        sequence_output, _ = self.mpnet(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        return logits


class MPNetForQuestionAnswering(MPNetPretrainedModel):
    def __init__(self, mpnet, num_classes=2):
        super(MPNetForQuestionAnswering, self).__init__()
        self.mpnet = mpnet
        self.num_classes = num_classes
        self.qa_outputs = nn.Linear(self.mpnet.config["hidden_size"],
                                    num_classes)

        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        sequence_output, _ = self.mpnet(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)
        logits = self.qa_outputs(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])

        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits
