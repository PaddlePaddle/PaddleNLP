# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle.nn as nn
import paddle
import paddle.nn.functional as F
from .. import PretrainedModel, register_base_model
import math

__all__ = [
    'LukeModel', 'LukePretrainedModel', 'LukeForEntitySpanClassification',
    'LukeForEntityPairClassification', 'LukeForEntityClassification',
    'LukeForMaskedLM', 'LukeForQuestionAnswering'
]


def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


def mish(x):
    return x * F.tanh(F.softplus(x))


def linear_act(x):
    return x


def swish(x):
    return x * F.sigmoid(x)


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": gelu_new,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": mish,
    "linear": linear_act,
    "swish": swish,
}


class LukePretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained Luke models. It provides Luke related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "luke-base": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "pad_token_id": 1,
            "entity_pad_token_id": -1,
            "layer_norm_eps": 1e-6,
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 514,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 1,
            "vocab_size": 50267
        },
        "luke-large": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "pad_token_id": 1,
            "entity_pad_token_id": -1,
            "layer_norm_eps": 1e-6,
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 514,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 1,
            "vocab_size": 50267
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "luke-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-base/model_state.pdparams",
            "luke-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-large/model_state.pdparams",
        }
    }
    base_model_prefix = "luke"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else self.luke.config[
                        "initializer_range"],
                    shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = self.layer_norm_eps if hasattr(
                self, "layer_norm_eps") else self.luke.config["layer_norm_eps"]


class LukeSelfOutput(nn.Layer):
    """Luke self output"""

    def __init__(self, hidden_size, layer_norm_eps, hidden_dropout_prob):
        super(LukeSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LukeIntermediate(nn.Layer):
    """Luke intermediate"""

    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super(LukeIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = get_activation(hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LukeOutput(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, layer_norm_eps,
                 hidden_dropout_prob):
        super(LukeOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LukeEmbeddings(nn.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self,
                 vocab_size=50267,
                 hidden_size=768,
                 max_position_embeddings=514,
                 type_vocab_size=1,
                 pad_token_id=0,
                 layer_norm_eps=1e-6,
                 hidden_dropout_prob=0.1):
        super(LukeEmbeddings, self).__init__()
        self.padding_idx = pad_token_id
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size, padding_idx=self.padding_idx)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = paddle.arange(
                self.padding_idx + 1,
                seq_length + self.padding_idx + 1,
                dtype='int64')
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        return self.fuse_embedding(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds)

    def fuse_embedding(
            self,
            input_ids,
            token_type_ids,
            position_ids,
            inputs_embeds, ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = paddle.arange(start=0, end=seq_length, dtype='int64')
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype='int64')

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LukePooler(nn.Layer):
    def __init__(self, hidden_size):
        super(LukePooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class EntityEmbeddings(nn.Layer):
    def __init__(self,
                 entity_vocab_size=500000,
                 entity_emb_size=256,
                 hidden_size=768,
                 max_position_embeddings=514,
                 type_vocab_size=1,
                 pad_token_id=0,
                 entity_pad_token_id=-1,
                 layer_norm_eps=1e-6,
                 hidden_dropout_prob=0.1):
        super(EntityEmbeddings, self).__init__()
        self.entity_emb_size = entity_emb_size
        self.hidden_size = hidden_size
        self.entity_pad_token_id = entity_pad_token_id
        self.entity_embeddings = nn.Embedding(
            entity_vocab_size, entity_emb_size, padding_idx=pad_token_id)
        if entity_emb_size != hidden_size:
            self.entity_embedding_dense = nn.Linear(
                entity_emb_size, hidden_size, bias_attr=False)

        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, entity_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.entity_emb_size != self.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(
            paddle.clip(
                position_ids, min=0))
        position_embedding_mask = (position_ids != self.entity_pad_token_id
                                   ).astype('float32').unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = paddle.sum(position_embeddings, axis=-2)
        position_embeddings = position_embeddings / paddle.clip(
            position_embedding_mask.sum(axis=-2), min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class EntityAwareSelfAttention(nn.Layer):
    def __init__(self,
                 num_attention_heads=12,
                 hidden_size=768,
                 attention_probs_dropout_prob=0.1):
        super(EntityAwareSelfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.w2e_query = nn.Linear(hidden_size, self.all_head_size)
        self.e2w_query = nn.Linear(hidden_size, self.all_head_size)
        self.e2e_query = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads, self.attention_head_size
        ]
        return paddle.transpose(x.reshape(new_x_shape), perm=(0, 2, 1, 3))

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_size = word_hidden_states.shape[1]

        w2w_query_layer = self.transpose_for_scores(
            self.query(word_hidden_states))
        w2e_query_layer = self.transpose_for_scores(
            self.w2e_query(word_hidden_states))
        e2w_query_layer = self.transpose_for_scores(
            self.e2w_query(entity_hidden_states))
        e2e_query_layer = self.transpose_for_scores(
            self.e2e_query(entity_hidden_states))

        key_layer = self.transpose_for_scores(
            self.key(
                paddle.concat(
                    [word_hidden_states, entity_hidden_states], axis=1)))

        w2w_key_layer = key_layer[:, :, :word_size, :]
        e2w_key_layer = key_layer[:, :, :word_size, :]
        w2e_key_layer = key_layer[:, :, word_size:, :]
        e2e_key_layer = key_layer[:, :, word_size:, :]

        w2w_attention_scores = paddle.matmul(
            w2w_query_layer, paddle.transpose(
                w2w_key_layer, perm=(0, 1, 3, 2)))
        w2e_attention_scores = paddle.matmul(
            w2e_query_layer, paddle.transpose(
                w2e_key_layer, perm=(0, 1, 3, 2)))
        e2w_attention_scores = paddle.matmul(
            e2w_query_layer, paddle.transpose(
                e2w_key_layer, perm=(0, 1, 3, 2)))
        e2e_attention_scores = paddle.matmul(
            e2e_query_layer, paddle.transpose(
                e2e_key_layer, perm=(0, 1, 3, 2)))

        word_attention_scores = paddle.concat(
            [w2w_attention_scores, w2e_attention_scores], axis=3)
        entity_attention_scores = paddle.concat(
            [e2w_attention_scores, e2e_attention_scores], axis=3)
        attention_scores = paddle.concat(
            [word_attention_scores, entity_attention_scores], axis=2)

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(
            self.value(
                paddle.concat(
                    [word_hidden_states, entity_hidden_states], axis=1)))
        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = paddle.transpose(context_layer, perm=(0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer[:, :
                             word_size, :], context_layer[:,
                                                          word_size:, :], attention_probs


class EntityAwareAttention(nn.Layer):
    def __init__(self,
                 hidden_size=768,
                 layer_norm_eps=1e-6,
                 hidden_dropout_prob=0.1,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1):
        super(EntityAwareAttention, self).__init__()
        self.self = EntityAwareSelfAttention(
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_size=hidden_size)
        self.output = LukeSelfOutput(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_self_output, entity_self_output, attention_probs = self.self(
            word_hidden_states, entity_hidden_states, attention_mask)
        hidden_states = paddle.concat(
            [word_hidden_states, entity_hidden_states], axis=1)
        self_output = paddle.concat(
            [word_self_output, entity_self_output], axis=1)
        output = self.output(self_output, hidden_states)
        return output[:, : word_hidden_states.shape[1], :], \
               output[:, word_hidden_states.shape[1]:, :], \
               attention_probs


class EntityAwareLayer(nn.Layer):
    def __init__(self,
                 hidden_size=768,
                 hidden_act="gelu",
                 intermediate_size=3072,
                 layer_norm_eps=1e-6,
                 hidden_dropout_prob=0.1,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1):
        super(EntityAwareLayer, self).__init__()

        self.attention = EntityAwareAttention(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob)
        self.intermediate = LukeIntermediate(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act)
        self.output = LukeOutput(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_attention_output, entity_attention_output, attention_probs = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask)
        attention_output = paddle.concat(
            [word_attention_output, entity_attention_output], axis=1)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output[:, : word_hidden_states.shape[1], :], \
               layer_output[:, word_hidden_states.shape[1]:, :], \
               attention_probs


class EntityAwareEncoder(nn.Layer):
    def __init__(self,
                 hidden_act,
                 num_hidden_layers=12,
                 hidden_size=768,
                 intermediate_size=3072,
                 layer_norm_eps=1e-6,
                 hidden_dropout_prob=0.1,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1):
        super(EntityAwareEncoder, self).__init__()
        self.layer = nn.LayerList([
            EntityAwareLayer(
                hidden_act=hidden_act,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_norm_eps=layer_norm_eps,
                hidden_dropout_prob=hidden_dropout_prob,
                num_attention_heads=num_attention_heads,
                attention_probs_dropout_prob=attention_probs_dropout_prob)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states, attention_probs = layer_module(
                word_hidden_states, entity_hidden_states, attention_mask)
        return word_hidden_states, entity_hidden_states, attention_probs


@register_base_model
class LukeModel(LukePretrainedModel):
    """   The bare LUKE model transformer outputting raw hidden-states for
          both word tokens and entities without any specific head on top.
    """

    def __init__(
            self,
            vocab_size=50265,
            pad_token_id=1,
            entity_pad_token_id=-1,
            initializer_range=0.02,
            max_position_embeddings=514,
            type_vocab_size=1,
            hidden_size=768,
            entity_vocab_size=500000,
            entity_emb_size=256,
            hidden_act="gelu",
            num_hidden_layers=12,
            intermediate_size=3072,
            layer_norm_eps=1e-6,
            hidden_dropout_prob=0.1,
            num_attention_heads=12,
            attention_probs_dropout_prob=0.1, ):
        super(LukeModel, self).__init__()
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder = EntityAwareEncoder(
            hidden_act=hidden_act,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob)
        self.embeddings = LukeEmbeddings(
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob)
        self.embeddings.token_type_embeddings.stop_gradient = True
        self.entity_embeddings = EntityEmbeddings(
            entity_pad_token_id=entity_pad_token_id,
            pad_token_id=pad_token_id,
            entity_vocab_size=entity_vocab_size,
            entity_emb_size=entity_emb_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob)
        self.pooler = LukePooler(hidden_size=hidden_size)
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids,
            token_type_ids,
            attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask, ):
        word_embeddings = self.embeddings(input_ids, token_type_ids)
        entity_embeddings = self.entity_embeddings(
            entity_ids, entity_position_ids, entity_segment_ids)
        _attention_mask = self._compute_extended_attention_mask(
            attention_mask, entity_attention_mask)
        word_hidden_state, entity_hidden_state, attention_probs = self.encoder(
            word_embeddings, entity_embeddings, _attention_mask)
        pool_output = self.pooler(word_hidden_state)
        return word_hidden_state, entity_hidden_state, pool_output, attention_probs

    def _compute_extended_attention_mask(self, word_attention_mask,
                                         entity_attention_mask):
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = paddle.concat(
                [attention_mask, entity_attention_mask], axis=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.astype('float32')
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class LukeLMHead(nn.Layer):
    """Luke Head for masked language modeling."""

    def __init__(self, vocab_size, hidden_size, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class EntityPredictionHeadTransform(nn.Layer):
    def __init__(self, hidden_act, hidden_size, entity_emb_size,
                 layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(hidden_size, entity_emb_size)
        self.transform_act_fn = get_activation(hidden_act)
        self.LayerNorm = nn.LayerNorm(entity_emb_size, epsilon=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class EntityPredictionHead(nn.Layer):
    def __init__(self, hidden_size, entity_vocab_size, entity_emb_size,
                 hidden_act, layer_norm_eps):
        super().__init__()
        self.transform = EntityPredictionHeadTransform(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            entity_emb_size=entity_emb_size,
            layer_norm_eps=layer_norm_eps)
        self.decoder = nn.Linear(entity_emb_size, entity_vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class LukeForMaskedLM(LukePretrainedModel):
    def __init__(self, luke):
        super().__init__()
        self.luke = luke
        self.vocab_size = self.luke.config['vocab_size']
        self.entity_vocab_size = self.luke.config['entity_vocab_size']

        self.lm_head = LukeLMHead(
            vocab_size=self.luke.config['vocab_size'],
            hidden_size=self.luke.config['hidden_size'],
            layer_norm_eps=self.luke.config['layer_norm_eps'])
        self.entity_predictions = EntityPredictionHead(
            hidden_size=self.luke.config['hidden_size'],
            hidden_act=self.luke.config['hidden_act'],
            entity_vocab_size=self.luke.config['entity_vocab_size'],
            entity_emb_size=self.luke.config['entity_emb_size'],
            layer_norm_eps=self.luke.config['layer_norm_eps'])

        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                entity_ids=None,
                entity_position_ids=None,
                entity_segment_ids=None,
                entity_attention_mask=None):
        outputs = self.luke(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask)

        logits = self.lm_head(outputs[0])
        entity_logits = self.entity_predictions(outputs[1])

        return logits, entity_logits


class LukeForEntityClassification(LukePretrainedModel):
    def __init__(self, luke, num_labels):
        super().__init__()

        self.luke = luke

        self.num_labels = num_labels
        self.dropout = nn.Dropout(self.luke.config['hidden_dropout_prob'])
        self.classifier = nn.Linear(self.luke.config['hidden_size'], num_labels)
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                entity_ids=None,
                entity_position_ids=None,
                entity_segment_ids=None,
                entity_attention_mask=None):
        outputs = self.luke(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask)

        feature_vector = outputs[1][:, 0, :]
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        return logits


class LukeForEntityPairClassification(LukePretrainedModel):
    def __init__(self, luke, num_labels):
        super().__init__()

        self.luke = luke

        self.num_labels = num_labels
        self.dropout = nn.Dropout(self.luke.config['hidden_dropout_prob'])
        self.classifier = nn.Linear(
            self.luke.config['hidden_size'] * 2, num_labels, bias_attr=False)
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            entity_ids=None,
            entity_position_ids=None,
            entity_segment_ids=None,
            entity_attention_mask=None, ):
        outputs = self.luke(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask)

        feature_vector = paddle.concat(
            [outputs[1][:, 0, :], outputs[1][:, 1, :]], axis=1)
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        return logits


class LukeForEntitySpanClassification(LukePretrainedModel):
    def __init__(self, luke, num_labels):
        super().__init__()

        self.luke = luke

        self.num_labels = num_labels
        self.dropout = nn.Dropout(self.luke.config['hidden_dropout_prob'])
        self.classifier = nn.Linear(self.luke.config['hidden_size'] * 3,
                                    num_labels)
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                entity_ids=None,
                entity_position_ids=None,
                entity_segment_ids=None,
                entity_attention_mask=None,
                entity_start_positions=None,
                entity_end_positions=None):
        outputs = self.luke(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask)
        hidden_size = outputs[0].shape[-1]

        entity_start_positions = entity_start_positions.unsqueeze(-1).expand(
            (-1, -1, hidden_size))
        start_states = paddle_gather(
            x=outputs[0], index=entity_start_positions, dim=-2)
        entity_end_positions = entity_end_positions.unsqueeze(-1).expand(
            (-1, -1, hidden_size))
        end_states = paddle_gather(
            x=outputs[0], index=entity_end_positions, dim=-2)
        feature_vector = paddle.concat(
            [start_states, end_states, outputs[1]], axis=2)

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        return logits


class LukeForQuestionAnswering(LukePretrainedModel):
    def __init__(self, luke):
        super(LukeForQuestionAnswering, self).__init__()
        self.luke = luke
        self.qa_outputs = nn.Linear(self.luke.config['hidden_size'], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                entity_ids=None,
                entity_position_ids=None,
                entity_segment_ids=None,
                entity_attention_mask=None):
        encoder_outputs = self.luke(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask)

        word_hidden_states = encoder_outputs[0][:, :input_ids.shape[1], :]
        logits = self.qa_outputs(word_hidden_states)
        start_logits, end_logits = paddle.split(logits, 2, -1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
