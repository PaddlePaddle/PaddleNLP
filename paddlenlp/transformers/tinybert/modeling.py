# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 Huawei Technologies Co., Ltd.
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

from ..bert.modeling import BertPooler, BertEmbeddings
from .. import PretrainedModel, register_base_model

__all__ = [
    'TinyBertModel',
    'TinyBertPretrainedModel',
    'TinyBertForPretraining',
    'TinyBertForSequenceClassification',
]


class TinyBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained TinyBERT models. It provides TinyBERT
    related `model_config_file`, `resource_files_names`,
    `pretrained_resource_files_map`, `pretrained_init_configuration`,
    `base_model_prefix` for downloading and loading pretrained models. See
    `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "tinybert-4l-312d": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-4l-312d-v2": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d-v2": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-4l-312d-zh": {
            "vocab_size": 21128,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d-zh": {
            "vocab_size": 21128,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "tinybert-4l-312d":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d.pdparams",
            "tinybert-6l-768d":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d.pdparams",
            "tinybert-4l-312d-v2":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d-v2.pdparams",
            "tinybert-6l-768d-v2":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d-v2.pdparams",
            "tinybert-4l-312d-zh":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d-zh.pdparams",
            "tinybert-6l-768d-zh":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d-zh.pdparams",
        }
    }
    base_model_prefix = "tinybert"

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
                        self.tinybert.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class TinyBertModel(TinyBertPretrainedModel):
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
                 pad_token_id=0,
                 fit_size=768):
        super(TinyBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = BertPooler(hidden_size)
        # fit_dense(s) means a hidden states' transformation from student to teacher.
        # `fit_denses` is used in v2 model, and `fit_dense` is used in other pretraining models.
        self.fit_denses = nn.LayerList([
            nn.Linear(hidden_size, fit_size)
            for i in range(num_hidden_layers + 1)
        ])
        self.fit_dense = nn.Linear(hidden_size, fit_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layer = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoded_layer)

        return encoded_layer, pooled_output


class TinyBertForPretraining(TinyBertPretrainedModel):
    def __init__(self, tinybert):
        super(TinyBertForPretraining, self).__init__()
        self.tinybert = tinybert
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.tinybert(
            input_ids, token_type_ids, attention_mask)

        return sequence_output


class TinyBertForSequenceClassification(TinyBertPretrainedModel):
    def __init__(self, tinybert, num_classes=2, dropout=None):
        super(TinyBertForSequenceClassification, self).__init__()
        self.tinybert = tinybert
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.tinybert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.tinybert.config["hidden_size"],
                                    num_classes)
        self.activation = nn.ReLU()
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.tinybert(
            input_ids, token_type_ids, attention_mask)

        logits = self.classifier(self.activation(pooled_output))
        return logits
