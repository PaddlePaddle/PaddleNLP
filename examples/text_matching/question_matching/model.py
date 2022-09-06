# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F

import paddlenlp as ppnlp


class QuestionMatching(nn.Layer):

    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):

        _, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,
                                     attention_mask)
        cls_embedding1 = self.dropout(cls_embedding1)
        logits1 = self.classifier(cls_embedding1)

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids,
                                         position_ids, attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return logits1, kl_loss
