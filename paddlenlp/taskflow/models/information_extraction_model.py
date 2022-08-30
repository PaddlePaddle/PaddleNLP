# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErniePretrainedModel


class UIE(ErniePretrainedModel):

    def __init__(self, encoding_model):
        super(UIE, self).__init__()
        self.encoder = encoding_model
        hidden_size = self.encoder.config["hidden_size"]
        self.linear_start = paddle.nn.Linear(hidden_size, 1)
        self.linear_end = paddle.nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, pos_ids, att_mask):
        sequence_output, pooled_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=att_mask)
        start_logits = self.linear_start(sequence_output)
        start_logits = paddle.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = paddle.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob
