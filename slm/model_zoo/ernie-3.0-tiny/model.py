# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn
from paddle.nn import Layer

from paddlenlp.transformers import ErnieModel, ErniePretrainedModel


class JointErnie(ErniePretrainedModel):
    def __init__(self, config, intent_dim, slot_dim, dropout=None):
        super(JointErnie, self).__init__(config)
        self.intent_num_labels = intent_dim
        self.slot_num_labels = slot_dim

        self.ernie = ErnieModel(config)
        self.dropout = nn.Dropout(dropout if dropout is not None else config["hidden_dropout_prob"])

        self.intent_classifier = nn.Linear(
            config["hidden_size"],
            self.intent_num_labels,
            weight_attr=nn.initializer.KaimingNormal(),
            bias_attr=nn.initializer.KaimingNormal(),
        )
        self.slot_classifier = nn.Linear(
            config["hidden_size"],
            self.slot_num_labels,
            weight_attr=nn.initializer.KaimingNormal(),
            bias_attr=nn.initializer.KaimingNormal(),
        )

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.ernie(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        if paddle.in_dynamic_mode():
            padding_mask = input_ids == 0
            padding_mask |= (input_ids == 2) | (input_ids == 1)
            return intent_logits, slot_logits, padding_mask

        return intent_logits * 1.0, slot_logits * 1.0


class NLULoss(Layer):
    def __init__(self, ignore_index=0):
        super(NLULoss, self).__init__()
        self.intent_loss_fct = paddle.nn.CrossEntropyLoss()
        self.slot_loss_fct = paddle.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        intent_label, slot_label = labels
        (
            intent_logits,
            slot_logits,
            _,
        ) = logits
        intent_loss = self.intent_loss_fct(intent_logits, intent_label)
        slot_loss = self.slot_loss_fct(slot_logits, slot_label)
        return slot_loss + intent_loss
