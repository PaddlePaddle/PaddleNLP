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


class PairwiseMatching(nn.Layer):

    def __init__(self, pretrained_model, dropout=None, margin=0.1):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.margin = margin

        # hidden_size -> 1, calculate similarity
        self.similarity = nn.Linear(self.ptm.config["hidden_size"], 1)

    @paddle.jit.to_static(input_spec=[
        paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        paddle.static.InputSpec(shape=[None, None], dtype='int64')
    ])
    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding)
        sim_score = self.similarity(cls_embedding)
        sim_score = F.sigmoid(sim_score)

        return sim_score

    def forward(self,
                pos_input_ids,
                neg_input_ids,
                pos_token_type_ids=None,
                neg_token_type_ids=None,
                pos_position_ids=None,
                neg_position_ids=None,
                pos_attention_mask=None,
                neg_attention_mask=None):

        _, pos_cls_embedding = self.ptm(pos_input_ids, pos_token_type_ids,
                                        pos_position_ids, pos_attention_mask)

        _, neg_cls_embedding = self.ptm(neg_input_ids, neg_token_type_ids,
                                        neg_position_ids, neg_attention_mask)

        pos_embedding = self.dropout(pos_cls_embedding)
        neg_embedding = self.dropout(neg_cls_embedding)

        pos_sim = self.similarity(pos_embedding)
        neg_sim = self.similarity(neg_embedding)

        pos_sim = F.sigmoid(pos_sim)
        neg_sim = F.sigmoid(neg_sim)

        labels = paddle.full(shape=[pos_cls_embedding.shape[0]],
                             fill_value=1.0,
                             dtype='float32')

        loss = F.margin_ranking_loss(pos_sim,
                                     neg_sim,
                                     labels,
                                     margin=self.margin)

        return loss
