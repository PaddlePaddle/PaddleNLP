# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn

from .. import PretrainedModel, register_base_model

__all__ = [
    'RobertaModel',
    'RobertaPretrainedModel',
    'RobertaForSequenceClassification',
    'RobertaForTokenClassification',
    'RobertaForQuestionAnswering',
]


class RobertaEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 pad_token_id=0):
        super(RobertaEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaPooler(nn.Layer):
    """
    """

    def __init__(self, hidden_size):
        super(RobertaPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained RoBERTa models. It provides RoBERTa related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "roberta-wwm-ext": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0
        },
        "roberta-wwm-ext-large": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0
        },
        "rbt3": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 3,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0,
        },
        "rbtl3": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 3,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "roberta-wwm-ext":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roberta_base/roberta_chn_base.pdparams",
            "roberta-wwm-ext-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roberta_large/roberta_chn_large.pdparams",
            "rbt3":
            "https://paddlenlp.bj.bcebos.com/models/transformers/rbt3/rbt3_chn_large.pdparams",
            "rbtl3":
            "https://paddlenlp.bj.bcebos.com/models/transformers/rbtl3/rbtl3_chn_large.pdparams",
        }
    }
    base_model_prefix = "roberta"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.roberta.config["initializer_range"],
                    shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class RobertaModel(RobertaPretrainedModel):
    """
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0):
        super(RobertaModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = RobertaEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = RobertaPooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class RobertaForQuestionAnswering(RobertaPretrainedModel):
    def __init__(self, roberta, dropout=None):
        super(RobertaForQuestionAnswering, self).__init__()
        self.roberta = roberta  # allow roberta to be config
        self.classifier = nn.Linear(self.roberta.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None):
        sequence_output, _ = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            attention_mask=None)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class RobertaForSequenceClassification(RobertaPretrainedModel):
    """
    Model for sentence (pair) classification task with RoBERTa.
    Args:
        roberta (RobertaModel): An instance of `RobertaModel`.
        num_classes (int, optional): The number of classes. Default 2
        dropout (float, optional): The dropout probability for output of RoBERTa.
            If None, use the same value as `hidden_dropout_prob` of `RobertaModel`
            instance `Roberta`. Default None
    """

    def __init__(self, roberta, num_classes=2, dropout=None):
        super(RobertaForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.roberta = roberta  # allow roberta to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roberta.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roberta.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, pooled_output = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RobertaForTokenClassification(RobertaPretrainedModel):
    def __init__(self, roberta, num_classes=2, dropout=None):
        super(RobertaForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.roberta = roberta  # allow roberta to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roberta.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roberta.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _ = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
