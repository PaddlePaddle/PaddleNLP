#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import traceback
import logging
import math

import numpy as np
import paddle
import paddle.nn.functional as F


def maybe_mask(attn, attn_mask):
    if attn_mask is not None:
        assert all(
            a == 1 or b == 1 or a == b
            for a, b in zip(attn.shape[::-1], attn_mask.shape[::-1])), \
            f'Attention mask shape {attn_mask.shape} should be broadcastable with attention shape {attn.shape}'

        attn.data.masked_fill_(attn_mask, -float('inf'))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.shape[-1]
    scores = paddle.matmul(query, key, transpose_y=True) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return paddle.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
    return paddle.matmul(p_attn, value), p_attn


class Attention(paddle.nn.Layer):

    def __init__(self, pointer):
        super().__init__()
        self.pointer = pointer
        self.softmax = paddle.nn.Softmax(axis=-1)

    def forward(self, query, values, attn_mask=None):
        # query shape: batch x query_size
        # values shape: batch x num values x value_size

        # attn_logits shape: batch x num values
        attn_logits = self.pointer(query, values, attn_mask)
        # attn_logits shape: batch x num values
        attn = self.softmax(attn_logits)
        # output shape: batch x 1 x value_size
        output = paddle.bmm(attn.unsqueeze(1), values)
        output = output.squeeze(1)
        return output, attn


class ScaledDotProductPointer(paddle.nn.Layer):

    def __init__(self, query_size, key_size):
        super().__init__()
        self.query_proj = paddle.nn.Linear(query_size, key_size)
        self.temp = np.power(key_size, 0.5)

    def forward(self, query, keys, attn_mask=None):
        # query shape: batch x query_size
        # keys shape: batch x num keys x key_size

        # proj_query shape: batch x key_size x 1
        proj_query = self.query_proj(query).unsqueeze(2)

        # attn_logits shape: batch x num keys
        attn_logits = paddle.bmm(keys, proj_query).squeeze(2) / self.temp
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


class ScaledDotProductAttention(Attention):

    def __init__(self, query_size, value_size):
        super().__init__(ScaledDotProductPointer(query_size, value_size))


class BahdanauPointer(paddle.nn.Layer):

    def __init__(self, query_size, key_size, proj_size):
        super().__init__()
        self.compute_scores = paddle.nn.Sequential(
            paddle.nn.Linear(query_size + key_size, proj_size),
            paddle.nn.Tanh(), paddle.nn.Linear(proj_size, 1))

    def forward(self,
                query: paddle.Tensor,
                keys: paddle.Tensor,
                attn_mask=None):
        # query shape: batch x query_size
        # keys shape: batch x num keys x key_size

        # query_expanded shape: batch x num keys x query_size
        query_expanded = query.unsqueeze(1).expand(
            [query.shape[0], keys.shape[1], query.shape[-1]])

        # scores shape: batch x num keys x 1
        attn_logits = self.compute_scores(
            # shape: batch x num keys x query_size + key_size
            paddle.concat((query_expanded, keys), axis=2))
        # scores shape: batch x num keys
        attn_logits = attn_logits.squeeze(2)
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


class BahdanauAttention(Attention):

    def __init__(self, query_size, value_size, proj_size):
        super().__init__(BahdanauPointer(query_size, value_size, proj_size))


# Adapted from The Annotated Transformers
class MultiHeadedAttention(paddle.nn.Layer):

    def __init__(self, h, query_size, value_size, dropout=0.1):
        super().__init__()
        assert query_size % h == 0
        assert value_size % h == 0

        # We assume d_v always equals d_k
        self.d_k = value_size // h
        self.h = h

        self.linears = paddle.nn.LayerList([
            paddle.nn.Linear(query_size, value_size),
            paddle.nn.Linear(value_size, value_size),
            paddle.nn.Linear(value_size, value_size),
            paddle.nn.Linear(value_size, value_size),
        ])

        self.attn = None
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, query, values, attn_mask=None):
        "Implements Figure 2"
        if attn_mask is not None:
            # Same mask applied to all h heads.
            attn_mask = attn_mask.unsqueeze(1)
        nbatches = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, keys, values = \
            [l(x).reshape([nbatches, -1, self.h, self.d_k]).transpose([0, 2, 1, 3])
             for l, x in zip(self.linears, (query, values, values))]

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = transformer.sparse_attention(
        x, self.attn = attention(query,
                                 keys,
                                 values,
                                 mask=attn_mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose([0, 2, 1, 3]).reshape([nbatches, -1, self.h * self.d_k])
        x = x.squeeze(1)
        return self.linears[3](x), self.attn


if __name__ == "__main__":
    """run some simple test cases"""
    sdpp = ScaledDotProductPointer(query_size=8, key_size=16)
    sdpa = ScaledDotProductAttention(query_size=8, value_size=16)
    bp = BahdanauPointer(query_size=8, key_size=16, proj_size=12)
    ba = BahdanauAttention(query_size=8, value_size=16, proj_size=12)
    mha = MultiHeadedAttention(h=2, query_size=8, value_size=16)

    q = paddle.to_tensor(list(range(1, 9)), dtype='float32').reshape([1, 8])
    v = paddle.to_tensor(list(range(1, 17)),
                         dtype='float32').reshape([1, 1, 16])

    print(sdpp(q, v))
    print(sdpa(q, v))
    print(bp(q, v))
    print(ba(q, v))
    print(mha(q, v))
