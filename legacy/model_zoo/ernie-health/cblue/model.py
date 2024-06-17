# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import ElectraConfig, ElectraModel, ElectraPretrainedModel


class ElectraForBinaryTokenClassification(ElectraPretrainedModel):
    """
    Electra Model with two linear layers on top of the hidden-states output layers,
    designed for token classification tasks with nesting.

    Args:
        electra (:class:`ElectraModel`):
            An instance of ElectraModel.
        num_classes (list):
            The number of classes.
        dropout (float, optionl):
            The dropout probability for output of Electra.
            If None, use the same value as `hidden_dropout_prob' of 'ElectraModel`
            instance `electra`. Defaults to None.
    """

    def __init__(self, config: ElectraConfig, num_classes_oth, num_classes_sym):
        super(ElectraForBinaryTokenClassification, self).__init__(config)
        self.num_classes_oth = num_classes_oth
        self.num_classes_sym = num_classes_sym
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_oth = nn.Linear(config.hidden_size, self.num_classes_oth)
        self.classifier_sym = nn.Linear(config.hidden_size, self.num_classes_sym)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        sequence_output = self.electra(input_ids, token_type_ids, position_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)

        logits_sym = self.classifier_sym(sequence_output)
        logits_oth = self.classifier_oth(sequence_output)

        return logits_oth, logits_sym


class MultiHeadAttentionForSPO(nn.Layer):
    """
    Multi-head attention layer for SPO task.
    """

    def __init__(self, embed_dim, num_heads, scale_value=768):
        super(MultiHeadAttentionForSPO, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale_value = scale_value**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim * num_heads)
        self.k_proj = nn.Linear(embed_dim, embed_dim * num_heads)

    def forward(self, query, key):
        q = self.q_proj(query)
        k = self.k_proj(key)
        q = paddle.reshape(q, shape=[0, 0, self.num_heads, self.embed_dim])
        k = paddle.reshape(k, shape=[0, 0, self.num_heads, self.embed_dim])
        q = paddle.transpose(q, perm=[0, 2, 1, 3])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        scores = paddle.matmul(q, k, transpose_y=True)
        scores = paddle.scale(scores, scale=self.scale_value)
        return scores


class ElectraForSPO(ElectraPretrainedModel):
    """
    Electra Model with a linear layer on top of the hidden-states output
    layers for entity recognition, and a multi-head attention layer for
    relation classification.

    Args:
        electra (:class:`ElectraModel`):
            An instance of ElectraModel.
        num_classes (int):
            The number of classes.
        dropout (float, optionl):
            The dropout probability for output of Electra.
            If None, use the same value as `hidden_dropout_prob' of 'ElectraModel`
            instance `electra`. Defaults to None.
    """

    def __init__(self, config: ElectraConfig):
        super(ElectraForSPO, self).__init__(config)
        self.num_classes = config.num_labels
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.span_attention = MultiHeadAttentionForSPO(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        outputs = self.electra(
            input_ids, token_type_ids, position_ids, attention_mask, output_hidden_states=True, return_dict=True
        )
        sequence_outputs = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states
        sequence_outputs = self.dropout(sequence_outputs)
        ent_logits = self.classifier(sequence_outputs)

        subject_output = all_hidden_states[-2]
        cls_output = paddle.unsqueeze(sequence_outputs[:, 0, :], axis=1)
        subject_output = subject_output + cls_output

        output_size = self.num_classes + self.electra.config["hidden_size"]  # noqa:F841
        rel_logits = self.span_attention(sequence_outputs, subject_output)

        return ent_logits, rel_logits
