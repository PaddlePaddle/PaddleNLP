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

import abc
import sys

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DualEncoder(nn.Layer):

    def __init__(self,
                 pretrained_model,
                 dropout=None,
                 output_emb_size=None,
                 use_cross_batch=False):
        super().__init__()
        self.ernie = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # if output_emb_size is not None, then add Linear layer to reduce embedding_size,
        # we recommend set output_emb_size = 256 considering the trade-off beteween
        # recall performance and efficiency

        self.output_emb_size = output_emb_size
        if output_emb_size is not None and output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(768,
                                                      output_emb_size,
                                                      weight_attr=weight_attr)

        self.use_cross_batch = use_cross_batch
        if self.use_cross_batch:
            self.rank = paddle.distributed.get_rank()
        else:
            self.rank = 0

    def get_cls_output(self,
                       input_ids,
                       token_type_ids=None,
                       position_ids=None,
                       attention_mask=None):
        sequence_output, cls_embedding = self.ernie(input_ids, token_type_ids,
                                                    position_ids,
                                                    attention_mask)
        cls_embedding = sequence_output[:, 0]
        return cls_embedding

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None):
        sequence_output, cls_embedding = self.ernie(input_ids, token_type_ids,
                                                    position_ids,
                                                    attention_mask)

        if self.output_emb_size is not None and self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        return cls_embedding

    def get_semantic_embedding(self, data_loader):
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data

                text_embeddings = self.get_cls_output(
                    input_ids, token_type_ids=token_type_ids)

                yield text_embeddings

    def cosine_sim(self,
                   query_input_ids,
                   title_input_ids,
                   query_token_type_ids=None,
                   query_position_ids=None,
                   query_attention_mask=None,
                   title_token_type_ids=None,
                   title_position_ids=None,
                   title_attention_mask=None):

        query_cls_embedding = self.get_cls_output(query_input_ids,
                                                  query_token_type_ids,
                                                  query_position_ids,
                                                  query_attention_mask)

        title_cls_embedding = self.get_cls_output(title_input_ids,
                                                  title_token_type_ids,
                                                  title_position_ids,
                                                  title_attention_mask)

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding,
                                axis=-1)
        return cosine_sim

    def forward(self,
                query_input_ids,
                pos_title_input_ids,
                neg_title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                pos_title_token_type_ids=None,
                pos_title_position_ids=None,
                pos_title_attention_mask=None,
                neg_title_token_type_ids=None,
                neg_title_position_ids=None,
                neg_title_attention_mask=None):

        query_cls_embedding = self.get_cls_output(query_input_ids,
                                                  query_token_type_ids,
                                                  query_position_ids,
                                                  query_attention_mask)

        pos_title_cls_embedding = self.get_cls_output(pos_title_input_ids,
                                                      pos_title_token_type_ids,
                                                      pos_title_position_ids,
                                                      pos_title_attention_mask)

        neg_title_cls_embedding = self.get_cls_output(neg_title_input_ids,
                                                      neg_title_token_type_ids,
                                                      neg_title_position_ids,
                                                      neg_title_attention_mask)

        all_title_cls_embedding = paddle.concat(
            x=[pos_title_cls_embedding, neg_title_cls_embedding], axis=0)

        if self.use_cross_batch:
            tensor_list = []
            paddle.distributed.all_gather(tensor_list, all_title_cls_embedding)
            all_title_cls_embedding = paddle.concat(x=tensor_list, axis=0)

        # multiply
        logits = paddle.matmul(query_cls_embedding,
                               all_title_cls_embedding,
                               transpose_y=True)

        batch_size = query_cls_embedding.shape[0]

        labels = paddle.arange(batch_size * self.rank * 2,
                               batch_size * (self.rank * 2 + 1),
                               dtype='int64')
        labels = paddle.reshape(labels, shape=[-1, 1])

        accuracy = paddle.metric.accuracy(input=logits, label=labels)
        loss = F.cross_entropy(input=logits, label=labels)

        return loss, accuracy
