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
from paddlenlp.transformers import BertPretrainedModel

class BertForMultiLabelClassifier(BertPretrainedModel):
    """
    Model for multi-label text classification task with BERT.
    Args:
        bert (BertModel, nn.Layer): An instance of BertModel.
        num_labels (int, optional): The number of labels. Default 2
        dropout (float, optional): The dropout probability for output of BERT.
            If None, use the same value as `hidden_dropout_prob` of `BertModel`
            instance `bert`. Default None
    """

    def __init__(self, bert, num_labels=2, dropout=None):
        super(BertForMultiLabelClassifier, self).__init__()
        self.num_labels = num_labels
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"],
                                    num_labels)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
