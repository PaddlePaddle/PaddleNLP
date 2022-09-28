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
import paddle.nn as nn
import paddle.nn.functional as F


class ErnieDocForTextMatching(nn.Layer):

    def __init__(self, ernie_doc, num_classes=2, dropout=None):
        super().__init__()
        self.ernie_doc = ernie_doc
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(ernie_doc.config["hidden_size"],
                                    num_classes)

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_memories,
                title_memories,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):

        _, query_pooled_output, query_mem, = self.ernie_doc(
            query_input_ids, query_memories, query_token_type_ids,
            query_position_ids, query_attention_mask)

        _, title_pooled_output, title_mem = self.ernie_doc(
            title_input_ids, title_memories, title_token_type_ids,
            title_position_ids, title_attention_mask)

        diff_pooled_output = query_pooled_output - title_pooled_output
        diff_pooled_output = self.dropout(diff_pooled_output)
        output = self.classifier(diff_pooled_output)
        return output, query_mem, title_mem
