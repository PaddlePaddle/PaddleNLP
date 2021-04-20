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

from paddlenlp.transformers import BertPretrainedModel, register_base_model

__all__ = [
    'DistilBertModel',
    'DistilBertForSequenceClassification',
    'DistilBertForTokenClassification',
    'DistilBertForQuestionAnswering',
    'DistilBertForMaskedLM',
]


class BertEmbeddings(nn.Layer):
    """
    Includes embeddings from word, position and does not include
    token_type embeddings.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.0,
                 max_position_embeddings=512,
                 type_vocab_size=16):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            position_ids.stop_gradient = True

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@register_base_model
class DistilBertModel(BertPretrainedModel):
    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,  #1,
            attention_probs_dropout_prob=0.0,  #1,
            max_position_embeddings=512,
            type_vocab_size=16,
            initializer_range=0.02,
            pad_token_id=0):
        super(DistilBertModel, self).__init__()
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
        self.apply(self.init_weights)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype("float32") * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(input_ids=input_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        return encoder_outputs


class DistilBertForSequenceClassification(BertPretrainedModel):
    def __init__(self, distilbert, num_classes=2, dropout=None):
        super(DistilBertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = distilbert  # allow bert to be config
        self.pre_classifier = nn.Linear(self.bert.config["hidden_size"],
                                        self.bert.config["hidden_size"])
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, attention_mask=None):
        distilbert_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = distilbert_output[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        import pdb
        pdb.set_trace()

        return logits


class DistilBertForQuestionAnswering(BertPretrainedModel):
    def __init__(self, distilbert, dropout=None):
        super(DistilBertForQuestionAnswering, self).__init__()
        self.bert = distilbert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, attention_mask=None):
        sequence_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        return start_logits, end_logits


class DistilBertForTokenClassification(BertPretrainedModel):
    def __init__(self, distilbert, num_classes=2, dropout=None):
        super(DistilBertForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = distilbert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, attention_mask=None):
        sequence_output = self.bert(input_ids, attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        import pdb
        pdb.set_trace()
        return logits


class DistilBertForMaskedLM(BertPretrainedModel):
    def __init__(self, distilbert):
        super(DistilBertForMaskedLM, self).__init__()
        self.bert = distilbert
        self.vocab_transform = nn.Linear(self.bert.config["hidden_size"],
                                         self.bert.config["hidden_size"])
        self.vocab_layer_norm = nn.LayerNorm(self.bert.config["hidden_size"])
        self.vocab_projector = nn.Linear(self.bert.config["hidden_size"],
                                         self.bert.config["vocab_size"])

        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None, ):
        distilbert_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask)
        prediction_logits = self.vocab_transform(distilbert_output)
        prediction_logits = nn.GELU()(prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)
        prediction_logits = self.vocab_projector(prediction_logits)

        return prediction_logits
