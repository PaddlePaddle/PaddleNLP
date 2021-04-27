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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SemanticIndexANCE(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, margin=0.3):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.margin = margin

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)

        return cls_embedding

    def forward(self,
                text_input_ids,
                pos_sample_input_ids,
                neg_sample_input_ids,
                text_token_type_ids=None,
                text_position_ids=None,
                text_attention_mask=None,
                pos_sample_token_type_ids=None,
                pos_sample_position_ids=None,
                pos_sample_attention_mask=None,
                neg_sample_token_type_ids=None,
                neg_sample_position_ids=None,
                neg_sample_attention_mask=None):

        text_cls_embedding = self.get_pooled_embedding(
            text_input_ids, text_token_type_ids, text_position_ids,
            text_attention_mask)

        pos_sample_cls_embedding = self.get_pooled_embedding(
            pos_sample_input_ids, pos_sample_token_type_ids,
            pos_sample_position_ids, pos_sample_attention_mask)

        neg_sample_cls_embedding = self.get_pooled_embedding(
            neg_sample_input_ids, neg_sample_token_type_ids,
            neg_sample_position_ids, neg_sample_attention_mask)

        pos_sample_sim = paddle.sum(text_cls_embedding *
                                    pos_sample_cls_embedding,
                                    axis=-1)
        global_neg_sample_sim = paddle.sum(text_cls_embedding *
                                           neg_sample_cls_embedding,
                                           axis=-1)

        labels = paddle.full(
            shape=[text_cls_embedding.shape[0]],
            fill_value=1.0,
            dtype='float32')

        loss = F.margin_ranking_loss(
            pos_sample_sim, global_neg_sample_sim, labels, margin=self.margin)

        return loss
