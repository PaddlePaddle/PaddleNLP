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


class SkepForTokenClassification(paddle.nn.Layer):
    def __init__(self, skep, num_classes=5, dropout=None):
        super(SkepForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.skep = skep
        self.dropout = paddle.nn.Dropout(
            dropout
            if dropout is not None else self.skep.config["hidden_dropout_prob"])
        self.classifier = paddle.nn.Linear(self.skep.config["hidden_size"],
                                           num_classes)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _ = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
