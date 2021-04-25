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


class SemanticIndexHardestNeg(nn.Layer):
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

    def cosine_sim(self,
                   query_input_ids,
                   title_input_ids,
                   query_token_type_ids=None,
                   query_position_ids=None,
                   query_attention_mask=None,
                   title_token_type_ids=None,
                   title_position_ids=None,
                   title_attention_mask=None):

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids,
            query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids,
            title_attention_mask)

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding,
                                axis=-1).numpy()
        return cosine_sim

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids,
            query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids,
            title_attention_mask)

        cosine_sim = paddle.matmul(
            query_cls_embedding, title_cls_embedding, transpose_y=True)
        pos_sim = paddle.diag(cosine_sim)

        # subtract 10000 from all diagnal elements of cosine_sim
        mask_socre = paddle.full(
            shape=[query_cls_embedding.shape[0]],
            fill_value=10000,
            dtype='float32')
        tmp_cosin_sim = cosine_sim - paddle.diag(mask_socre)
        hardest_negative_sim = paddle.max(tmp_cosin_sim, axis=-1)

        labels = paddle.full(
            shape=[query_cls_embedding.shape[0]],
            fill_value=1.0,
            dtype='float32')

        loss = F.margin_ranking_loss(
            pos_sim, hardest_negative_sim, labels, margin=self.margin)
        return loss
