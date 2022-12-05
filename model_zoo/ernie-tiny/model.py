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
from paddle import nn
from paddle.nn import Layer

from paddlenlp.transformers import ErniePretrainedModel


class JointErnie(ErniePretrainedModel):
    def __init__(self, ernie, intent_dim, slot_dim, dropout=None):
        super(JointErnie, self).__init__()
        self.intent_num_labels = intent_dim
        self.slot_num_labels = slot_dim

        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config["hidden_dropout_prob"])

        self.intent_classifier = nn.Linear(self.ernie.config["hidden_size"], self.intent_num_labels)
        self.slot_classifier = nn.Linear(self.ernie.config["hidden_size"], self.slot_num_labels)

        self.intent_hidden = nn.Linear(self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])
        self.slot_hidden = nn.Linear(self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])

        self.apply(self.init_weights)

    def forward(self, words_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.ernie(
            words_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))
        pooled_output = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return intent_logits, slot_logits


class NLULoss(Layer):
    def __init__(self, pos_weight):
        super(NLULoss, self).__init__()

        self.intent_loss_fn = paddle.nn.BCEWithLogitsLoss(pos_weight=paddle.to_tensor(pos_weight))
        self.slot_loss_fct = paddle.nn.CrossEntropyLoss()

    def forward(self, logits, slot_labels, intent_labels):
        slot_logits, intent_logits = logits

        slot_loss = self.slot_loss_fct(slot_logits, slot_labels)
        intent_loss = self.intent_loss_fn(intent_logits, intent_labels)

        return slot_loss + intent_loss
