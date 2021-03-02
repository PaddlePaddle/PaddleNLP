# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
import paddle.nn.functional as F
import paddle.nn as nn
from ..attention_utils import _convert_param_attr_to_list, MultiHeadAttention, \
        AttentionRegistry, Mask, create_bigbird_simulated_attention_mask_list, create_bigbird_rand_mask_idx_list
from .. import PretrainedModel, register_base_model


class Mask(object):
    def __init__(self,
                 query_length,
                 key_length,
                 num_heads,
                 block_size,
                 window_size,
                 num_global_blocks,
                 num_rand_blocks,
                 seed=None):
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)
        self.mask = np.zeros_like(
            np.arange(query_length * key_length * num_heads).reshape((
                num_heads, query_length, key_length)))
        self.rand_mask = np.zeros_like(
            np.arange(query_length * key_length * num_heads).reshape((
                num_heads, query_length, key_length)))
        self.rand_mask_idx = [[] for i in range(num_heads)]
        self.num_query_blocks = self.query_length // self.block_size     \
                + int(self.query_length % self.block_size != 0)
        self.num_key_blocks = self.key_length // self.block_size         \
                + int(self.key_length % self.block_size != 0)
        self.num_window_blocks = self.window_size // 2
        if seed:
            np.random.seed(seed)
        # create global mask
        self._create_global_mask()
        # create window mask
        self._create_window_mask()
        # create random mask
        self._create_random_mask()

    def get_mask(self):
        return self.mask

    def get_rand_mask_idx(self):
        return self.rand_mask_idx

    def get_rand_mask(self):
        return self.rand_mask

    def get_float_mask(self):
        float_mask = np.array(self.mask, dtype='float32')
        float_mask[float_mask != 1] = -np.inf
        float_mask[float_mask == 1.] = 0
        return float_mask

    def _create_global_mask(self):
        global_block_length = self.num_global_blocks * self.block_size
        self.mask[:, 0:global_block_length, :] = 1
        self.mask[:, :, 0:global_block_length] = 1

    def _create_window_mask(self):
        for query_block_idx in range(self.num_query_blocks):
            left_key_block_idx, right_key_block_idx = self._get_window_block_idx(
                query_block_idx)
            left_idx = left_key_block_idx * self.block_size
            right_idx = (right_key_block_idx + 1) * self.block_size
            query_left_idx = query_block_idx * self.block_size
            query_right_idx = min((query_block_idx + 1) * self.block_size,
                                  self.query_length)
            self.mask[:, query_left_idx:query_right_idx, left_idx:right_idx] = 1

    def _create_random_mask(self):
        all_key_blocks_idx = np.arange(0, self.num_key_blocks, dtype=np.int32)
        for query_block_idx in range(self.num_query_blocks):
            left_key_block_idx, right_key_block_idx = self._get_window_block_idx(
                query_block_idx)
            illegal_blocks_idx = [
                i for i in range(left_key_block_idx, right_key_block_idx + 1)
            ]
            illegal_blocks_idx.extend(
                [i for i in range(self.num_global_blocks)])
            left_key_block_idx = query_block_idx - self.num_window_blocks
            right_key_block_idx = query_block_idx + self.num_window_blocks
            if self.num_global_blocks > left_key_block_idx:
                num_fill_blocks = self.num_global_blocks - left_key_block_idx
                illegal_blocks_idx.extend([
                    i for i in range(self.num_key_blocks - num_fill_blocks,
                                     self.num_key_blocks)
                ])
            if right_key_block_idx >= self.num_key_blocks:
                num_fill_blocks = right_key_block_idx - self.num_key_blocks + 1
                illegal_blocks_idx.extend([
                    i for i in range(self.num_global_blocks,
                                     self.num_global_blocks + num_fill_blocks)
                ])

            illegal_blocks_idx = set(illegal_blocks_idx)

            query_left_idx = query_block_idx * self.block_size
            query_right_idx = min((query_block_idx + 1) * self.block_size,
                                  self.query_length)
            for i in range(self.num_heads):
                legal_blocks_idx = []
                legal_idx = []
                perm_block = np.random.permutation(all_key_blocks_idx)
                for j in perm_block:
                    if j not in illegal_blocks_idx:
                        legal_blocks_idx.append(j)
                    if len(legal_blocks_idx) == self.num_rand_blocks:
                        break
                for j in legal_blocks_idx:
                    key_left_idx = j * self.block_size
                    key_right_idx = min((j + 1) * self.block_size,
                                        self.key_length)
                    legal_idx.extend(
                        [i for i in range(key_left_idx, key_right_idx)])
                    self.rand_mask[i, query_left_idx:query_right_idx,
                                   key_left_idx:key_right_idx] = 1
                self.rand_mask_idx[i].append(legal_blocks_idx)
        self.rand_mask_idx = np.stack(self.rand_mask_idx, axis=0)
        self.rand_mask_idx = self.rand_mask_idx[:, self.num_global_blocks:]
        self.mask = np.maximum(self.rand_mask, self.mask)

    def _get_window_block_idx(self, query_block_idx):
        left_key_block_idx = max(0, query_block_idx - self.num_window_blocks)
        right_key_block_idx = min(query_block_idx + self.num_window_blocks,
                                  self.num_key_blocks - 1)
        return left_key_block_idx, right_key_block_idx


def create_bigbird_attention_mask_list(
        num_layers, query_length, key_length, num_heads, block_size,
        window_size, num_global_blocks, num_rand_blocks, seed):
    attn_mask_list = []
    rand_mask_idx_list = []
    for i in range(num_layers):
        mask = Mask(query_length, key_length, num_heads, block_size,
                    window_size, num_global_blocks, num_rand_blocks, seed)
        attn_mask = paddle.to_tensor(mask.get_float_mask())
        rand_mask_idx = paddle.to_tensor(mask.get_rand_mask_idx())
        attn_mask_list.append(attn_mask)
        rand_mask_idx_list.append(rand_mask_idx)
    return attn_mask_list, rand_mask_idx_list


class TransformerEncoderLayer(Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 attention_type="bigbird",
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            attention_type=attention_type,
            block_size=block_size,
            window_size=window_size,
            num_global_blocks=num_global_blocks,
            num_rand_blocks=num_rand_blocks,
            seed=seed)
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model, epsilon=1e-12)
        self.norm2 = LayerNorm(d_model, epsilon=1e-12)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self.d_model = d_model

    def forward(self,
                src,
                src_mask=None,
                rand_mask_idx=None,
                query_mask=None,
                key_mask=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src = self.self_attn(src, src, src, src_mask, rand_mask_idx, query_mask,
                             key_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(Layer):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = LayerNorm(self.layers[0].d_model, epsilon=1e-12)
        self.normalize_before = self.layers[0].normalize_before

    def forward(self,
                src,
                src_mask_list=None,
                rand_mask_idx_list=None,
                query_mask=None,
                key_mask=None):
        output = src
        if not self.normalize_before:
            output = self.norm(output)

        for i, mod in enumerate(self.layers):
            rand_mask_id = None
            if rand_mask_idx_list is not None:
                rand_mask_id = rand_mask_idx_list[i]
            if src_mask_list is None:
                output = mod(output, None, rand_mask_id, query_mask, key_mask)
            else:
                output = mod(output, src_mask_list[i], rand_mask_id, query_mask,
                             key_mask)

        if self.normalize_before:
            output = self.norm(output)
        return output


class BigBirdPooler(Layer):
    """
    """

    def __init__(self, hidden_size):
        super(BigBirdPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BigBirdEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16):
        super(BigBirdEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class BigBirdPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained BigBird models. It provides BigBird related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "bigbird-base-uncased": {
            "num_layers": 12,
            "vocab_size": 50358,
            "nhead": 12,
            "attn_dropout": 0.1,
            "dim_feedforward": 3072,
            "activation": "gelu",
            "normalize_before": False,
            "block_size": 16,
            "window_size": 3,
            "num_global_blocks": 2,
            "num_rand_blocks": 3,
            "seed": None,
            "pad_token_id": 0,
            "hidden_size": 768,
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 4096,
            "type_vocab_size": 2,
            "num_labels": 2,
            "initializer_range": 0.02,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "bigbird-base-uncased":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bigbird/bigbird-base-uncased.pdparams",
        }
    }
    base_model_prefix = "bigbird"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.bigbird.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class BigBirdModel(BigBirdPretrainedModel):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 nhead,
                 attn_dropout=0.1,
                 dim_feedforward=3072,
                 activation="gelu",
                 normalize_before=False,
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=2,
                 seed=None,
                 pad_token_id=0,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 **kwargs):
        super(BigBirdModel, self).__init__()
        # embedding
        self.embeddings = BigBirdEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)

        # encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_size,
            nhead,
            dim_feedforward,
            attn_dropout,
            activation,
            normalize_before=normalize_before,
            attention_type="bigbird",
            block_size=block_size,
            window_size=window_size,
            num_global_blocks=num_global_blocks,
            num_rand_blocks=num_rand_blocks,
            seed=seed)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        # pooler
        self.pooler = BigBirdPooler(hidden_size)
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers

    def _process_mask(self, input_ids, attention_mask_list=None):
        # [B, T]
        attention_mask = (input_ids == self.pad_token_id
                          ).astype(self.pooler.dense.weight.dtype)
        # [B, 1, T, 1]
        query_mask = paddle.unsqueeze(attention_mask, axis=[1, 3])
        # [B, 1, 1, T]
        key_mask = paddle.unsqueeze(attention_mask, axis=[1, 2])
        query_mask = 1 - query_mask
        key_mask = 1 - key_mask
        return attention_mask_list, query_mask, key_mask

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask_list=None,
                rand_mask_idx_list=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        attention_mask_list, query_mask, key_mask = self._process_mask(
            input_ids, attention_mask_list)
        encoder_output = self.encoder(embedding_output, attention_mask_list,
                                      rand_mask_idx_list, query_mask, key_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output


class BigBirdForTokenClassification(BigBirdPretrainedModel):
    def __init__(self, bigbird):
        super(BigBirdForTokenClassification, self).__init__()
        self.bigbird = bigbird
        self.linear = nn.Linear(self.bigbird.config["hidden_size"],
                                self.bigbird.config["num_labels"])
        self.dropout = nn.Dropout(
            self.bigbird.config['hidden_dropout_prob'], mode="upscale_in_train")
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                attention_mask_list=None,
                rand_mask_idx_list=None):
        _, pooled_output = self.bigbird(
            input_ids,
            None,
            attention_mask_list=attention_mask_list,
            rand_mask_idx_list=rand_mask_idx_list)
        output = self.dropout(pooled_output)
        output = self.linear(output)
        return output


class BigBirdLMPredictionHead(Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(BigBirdLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=1e-12)
        self.decoder_weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
            dtype=self.transform.weight.dtype,
            is_bias=True) if embedding_weights is None else embedding_weights
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


class BigBirdPretrainingHeads(Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(BigBirdPretrainingHeads, self).__init__()
        self.predictions = BigBirdLMPredictionHead(
            hidden_size, vocab_size, activation, embedding_weights)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BigBirdForPretraining(BigBirdPretrainedModel):
    def __init__(self, bigbird):
        super(BigBirdForPretraining, self).__init__()
        self.bigbird = bigbird
        self.cls = BigBirdPretrainingHeads(
            self.bigbird.config["hidden_size"],
            self.bigbird.config["vocab_size"],
            self.bigbird.config["activation"],
            embedding_weights=self.bigbird.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                rand_mask_idx_list=None,
                masked_positions=None):
        outputs = self.bigbird(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask_list=None,
            rand_mask_idx_list=rand_mask_idx_list)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output, masked_positions)
        return prediction_scores, seq_relationship_score


class BigBirdPretrainingCriterion(paddle.nn.Layer):
    def __init__(self, vocab_size, use_nsp=False):
        super(BigBirdPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        self.use_nsp = use_nsp

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels, masked_lm_scale,
                masked_lm_weights):
        masked_lm_loss = paddle.nn.functional.softmax_with_cross_entropy(
            prediction_scores, masked_lm_labels, ignore_index=-1)
        masked_lm_loss = paddle.transpose(masked_lm_loss, [1, 0])
        masked_lm_loss = paddle.sum(masked_lm_loss * masked_lm_weights) / (
            paddle.sum(masked_lm_weights) + 1e-5)
        if not self.use_nsp:
            return masked_lm_loss
        next_sentence_loss = paddle.nn.functional.softmax_with_cross_entropy(
            seq_relationship_score, next_sentence_labels)
        return masked_lm_loss + paddle.mean(next_sentence_loss)
