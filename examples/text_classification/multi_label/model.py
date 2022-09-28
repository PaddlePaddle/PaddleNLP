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

import paddle.nn as nn


class MultiLabelClassifier(nn.Layer):

    def __init__(self, pretrained_model, num_labels=2, dropout=None):
        super(MultiLabelClassifier, self).__init__()
        self.ptm = pretrained_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ptm.
                                  config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], num_labels)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, pooled_output = self.ptm(input_ids,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
