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
from paddle import nn
import paddle.nn.functional as F

ACT_DICT = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
}


def _build_linear(n_in, n_out, name=None, init=None):
    return nn.Linear(
        n_in,
        n_out,
        weight_attr=paddle.ParamAttr(name='%s.w_0' %
                                     name if name is not None else None,
                                     initializer=init),
        bias_attr='%s.b_0' % name if name is not None else None,
    )


def _build_ln(n_in, name):
    return nn.LayerNorm(
        normalized_shape=n_in,
        weight_attr=paddle.ParamAttr(name='%s_layer_norm_scale' %
                                     name if name is not None else None,
                                     initializer=nn.initializer.Constant(1.)),
        bias_attr=paddle.ParamAttr(name='%s_layer_norm_bias' %
                                   name if name is not None else None,
                                   initializer=nn.initializer.Constant(0.)),
    )


def new_name(name, postfix):
    if name is None:
        ret = None
    elif name == '':
        ret = postfix
    else:
        ret = '%s_%s' % (name, postfix)
    return ret


# Adapted from
# https://github.com/tensorflow/tensor2tensor/blob/0b156ac533ab53f65f44966381f6e147c7371eee/tensor2tensor/layers/common_attention.py
def relative_attention_logits(query, key, relation):
    """relative attention logits(scores)

    Args:
        query (TYPE): NULL
        key (TYPE): NULL
        relation (TYPE): NULL

    Returns: Tensor, shape = [batch, heads, num queries, num kvs]

    Raises: NULL
    """
    # We can't reuse the same logic as tensor2tensor because we don't share relation vectors across the batch.
    # In this version, relation vectors are shared across heads.
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # qk_matmul is [batch, heads, num queries, num kvs]
    qk_matmul = paddle.matmul(query, key, transpose_y=True)
    if relation is None:
        return qk_matmul / math.sqrt(query.shape[-1])

    # q_t is [batch, num queries, heads, depth]
    q_t = query.transpose([0, 2, 1, 3])
    # r_t is [batch, num queries, depth, num kvs]
    r_t = relation.transpose([0, 1, 3, 2])

    # [batch, num queries, heads, depth] * [batch, num queries, depth, num kvs]
    # = [batch, num queries, heads, num kvs]
    # For each batch and query, we have a query vector per head.
    # We take its dot product with the relation vector for each kv.
    #
    # transposed = [batch, heads, num queries, num kvs]
    qr_matmul = paddle.matmul(q_t, r_t).transpose([0, 2, 1, 3])

    # [batch, heads, num queries, num kvs]
    return (qk_matmul + qr_matmul) / math.sqrt(query.shape[-1])

    # Sharing relation vectors across batch and heads:
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [num queries, num kvs, depth].
    #
    # Then take
    # key reshaped
    #   [num queries, batch * heads, depth]
    # relation.transpose(-2, -1)
    #   [num queries, depth, num kvs]
    # and multiply them together.
    #
    # Without sharing relation vectors across heads:
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, heads, num queries, num kvs, depth].
    #
    # Then take
    # key.unsqueeze(3)
    #   [batch, heads, num queries, 1, depth]
    # relation.transpose(-2, -1)
    #   [batch, heads, num queries, depth, num kvs]
    # and multiply them together:
    #   [batch, heads, num queries, 1, depth]
    # * [batch, heads, num queries, depth, num kvs]
    # = [batch, heads, num queries, 1, num kvs]
    # and squeeze
    # [batch, heads, num queries, num kvs]


def relative_attention_values(weight, value, relation):
    """In this version, relation vectors are shared across heads.
    Args:
        weight: [batch, heads, num queries, num kvs].
        value: [batch, heads, num kvs, depth].
        relation: [batch, num queries, num kvs, depth].
    Returns: Tensor, shape = [batch, heads, num queries, depth]
    """
    # wv_matmul is [batch, heads, num queries, depth]
    wv_matmul = paddle.matmul(weight, value)

    # w_t is [batch, num queries, heads, num kvs]
    w_t = weight.transpose([0, 2, 1, 3])
    #   [batch, num queries, heads, num kvs]
    # * [batch, num queries, num kvs, depth]
    # = [batch, num queries, heads, depth]
    # transposed = [batch, heads, num queries, depth]
    wr_matmul = paddle.matmul(w_t, relation).transpose([0, 2, 1, 3])

    return wv_matmul + wr_matmul


class RelationalAttentionLayer(nn.Layer):

    def __init__(self, cfg, name=None):
        super(RelationalAttentionLayer, self).__init__()
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        n_head = cfg['num_attention_heads']
        assert d_model % n_head == 0
        d_model_q = cfg.get('query_hidden_size_per_head',
                            d_model // n_head) * n_head
        d_model_v = cfg.get('value_hidden_size_per_head',
                            d_model // n_head) * n_head
        self.n_head = n_head
        self.d_key = d_model_q // n_head
        self.q = _build_linear(d_model, d_model_q, new_name(name, 'query_fc'),
                               initializer)
        self.k = _build_linear(d_model, d_model_q, new_name(name, 'key_fc'),
                               initializer)
        self.v = _build_linear(d_model, d_model_v, new_name(name, 'value_fc'),
                               initializer)
        self.o = _build_linear(d_model_v, d_model, new_name(name, 'output_fc'),
                               initializer)
        self.dropout = nn.Dropout(p=cfg['attention_probs_dropout_prob'])

    def forward(self,
                queries,
                keys,
                values,
                relation_k,
                relation_v,
                attn_bias=None,
                past_cache=None):
        """relational attention forward.
        seq_len in `shape` means num queries/keys/values of attention

        Args:
            queries (TYPE): shape = [batch, seq_len, num_heads * hidden]
            keys (TYPE): shape = queries.shape
            values (TYPE): shape = queries.shape
            relation_k (TYPE): shape = [batch, seq_len, seq_len, hidden]
            relation_v (TYPE): shape = relation_k.shape
            attn_bias (TYPE): used as sequence mask. Default is None
            past_cache (TYPE): Default is None

        Returns: TODO

        Raises: NULL
        """
        assert len(queries.shape) == len(keys.shape) == len(values.shape) == 3
        #bsz, q_len, q_dim = queries.shape
        #bsz, k_len, k_dim = keys.shape
        #bsz, v_len, v_dim = values.shape
        #assert k_len == v_len

        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)

        cache = (k, v)
        if past_cache is not None:
            cached_k, cached_v = past_cache
            k = paddle.concat([cached_k, k], 1)
            v = paddle.concat([cached_v, v], 1)

        def _transpose(inputs):
            """reshape and transpose
            Args: inputs: shape = [batch, seq_len, heads * hidden]
            Returns: shape = [batch, heads, seq_len, hidden]
            """
            hidden_size = inputs.shape[-1] // self.n_head
            outputs = inputs.reshape([0, 0, self.n_head, hidden_size])
            return outputs.transpose([0, 2, 1, 3])

        q, k, v = [_transpose(x) for x in (q, k, v)]

        q = q.scale(self.d_key**-0.5)
        scores = relative_attention_logits(q, k, relation_k)
        if attn_bias is not None:
            scores += attn_bias
        scores = F.softmax(scores)
        scores = self.dropout(scores)

        out = relative_attention_values(scores, v, relation_v)
        # input: [batch, heads, seq_len, hidden]
        # output: [batch, seq_len, heads * hidden]
        out = out.transpose([0, 2, 1, 3])
        out = out.reshape([0, 0, out.shape[2] * out.shape[3]])
        out = self.o(out)
        return out, cache


class RelationalPointerNet(nn.Layer):
    """Pointer Netword with Relations"""

    def __init__(self, hidden_size, num_relations, init_range=0.02, name=None):
        """init of class

        Args:
            cfg (TYPE): NULL

        """
        super(RelationalPointerNet, self).__init__()
        self.hidden_size = hidden_size

        initializer = nn.initializer.TruncatedNormal(std=init_range)
        self.q = _build_linear(hidden_size, hidden_size,
                               new_name(name, 'query_fc'), initializer)
        self.k = _build_linear(hidden_size, hidden_size,
                               new_name(name, 'key_fc'), initializer)
        #self.dropout = nn.Dropout(p=cfg['attention_probs_dropout_prob'])

        self.relation_emb = None
        if num_relations > 0:
            self.relation_emb = nn.Embedding(num_relations, hidden_size)
        self.scores = None

    def forward(self, queries, keys, relations, attn_bias=None):
        """relational attention forward.
        seq_len in `shape` means num queries/keys/values of attention

        Args:
            queries (TYPE): shape = [batch, seq_len, num_heads * hidden]
            keys (TYPE): shape = queries.shape
            relations (TYPE): shape = [batch, seq_len, seq_len, hidden]
            attn_bias (TYPE): used as sequence mask. Default is None

        Returns: TODO

        Raises: NULL
        """
        assert len(queries.shape) == len(keys.shape) == 3

        q = self.q(queries)
        k = self.k(keys)
        r = None
        if relations is not None:
            r = self.relation_emb(relations)

        def _transpose(inputs):
            """reshape and transpose
            Args: inputs: shape = [batch, seq_len, heads * hidden]
            Returns: shape = [batch, heads, seq_len, hidden]
            """
            # 1 代表 head 数量，此处恒为 1。
            outputs = inputs.reshape([0, 0, 1, self.hidden_size])
            return outputs.transpose([0, 2, 1, 3])

        q = _transpose(q)
        k = _transpose(k)
        #q = q.scale(self.hidden_size**-0.5)
        scores = relative_attention_logits(q, k, r)
        if attn_bias is not None:
            scores += attn_bias

        self.scores = F.softmax(scores)
        return self.scores.squeeze([0, 1])


class PositionwiseFeedForwardLayer(nn.Layer):

    def __init__(self, cfg, name=None):
        super(PositionwiseFeedForwardLayer, self).__init__()
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_ffn = cfg.get('intermediate_size', 4 * d_model)
        self.act = ACT_DICT[cfg['hidden_act']]()
        self.i = _build_linear(
            d_model,
            d_ffn,
            new_name(name, 'fc_0'),
            initializer,
        )
        self.o = _build_linear(d_ffn, d_model, new_name(name, 'fc_1'),
                               initializer)
        prob = cfg.get('intermediate_dropout_prob', 0.)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, inputs):
        hidden = self.act(self.i(inputs))
        hidden = self.dropout(hidden)
        out = self.o(hidden)
        return out


class RelationalTransformerBlock(nn.Layer):
    """A transformer block with relations"""

    def __init__(self, cfg, name=None):
        super(RelationalTransformerBlock, self).__init__()
        d_model = cfg['hidden_size']
        n_heads = cfg['num_attention_heads']
        self.attn = RelationalAttentionLayer(cfg,
                                             name=new_name(
                                                 name, 'multi_head_att'))
        self.ln1 = _build_ln(d_model, name=new_name(name, 'post_att'))
        self.ffn = PositionwiseFeedForwardLayer(cfg, name=new_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name=new_name(name, 'post_ffn'))
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)

        # 假设 k/v 的
        rel_hidden = d_model // n_heads
        self.relation_k_emb = nn.Embedding(cfg['num_relations'], rel_hidden)
        self.relation_v_emb = nn.Embedding(cfg['num_relations'], rel_hidden)

    def forward(self, inputs, relations, attn_bias=None, past_cache=None):
        relation_k = self.relation_k_emb(relations)
        relation_v = self.relation_k_emb(relations)

        attn_out, cache = self.attn(inputs,
                                    inputs,
                                    inputs,
                                    relation_k,
                                    relation_v,
                                    attn_bias,
                                    past_cache=past_cache)  #self attn
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm

        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden, cache


class RelationalTransformerEncoder(nn.Layer):

    def __init__(self, cfg, name=None):
        super(RelationalTransformerEncoder, self).__init__()
        n_layers = cfg['num_hidden_layers']
        self.block = nn.LayerList([
            RelationalTransformerBlock(cfg, new_name(name, 'layer_%d' % i))
            for i in range(n_layers)
        ])

    def forward(self, inputs, relations, attn_bias=None, past_cache=None):
        """relational transformer encoder, forward stage of
        n layers and m heads transformer blocks with relations

        Args:
            inputs (TYPE): shape= [batch, seq_len, hidden]
            relations (TYPE): shape = [batch, seq_len, seq_len]
            attn_bias (TYPE): mask for inputs sequence. Default is None
            past_cache (TYPE): Default is None

        Returns: (last_hidden_state, all_hidden_state_list, (cache_list_k, cache_list_v))

        Raises: NULL
        """
        if past_cache is not None:
            assert isinstance(past_cache, tuple), 'unknown type of `past_cache`,' + \
                   ' expect tuple or list. got %s' % repr(type(past_cache))
            past_cache = list(zip(*past_cache))
        else:
            past_cache = [None] * len(self.block)
        cache_list_k, cache_list_v, hidden_list = [], [], [inputs]

        for b, p in zip(self.block, past_cache):
            inputs, cache = b(inputs,
                              relations,
                              attn_bias=attn_bias,
                              past_cache=p)
            cache_k, cache_v = cache
            cache_list_k.append(cache_k)
            cache_list_v.append(cache_v)
            hidden_list.append(inputs)

        return inputs, hidden_list, (cache_list_k, cache_list_v)


if __name__ == "__main__":
    """run some simple test cases"""
    cfg = {
        "num_hidden_layers": 12,
        "num_attention_heads": 2,
        "num_relations": 99,
        "hidden_size": 4,
        "hidden_act": "relu",
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
    }

    model = RelationalTransformerEncoder(cfg)
    print(model)
    inputs = paddle.to_tensor(list(range(24)),
                              dtype='float32').reshape([2, 3, 4])
    relations = paddle.to_tensor(list(range(18)),
                                 dtype='int64').reshape([2, 3, 3])
    hidden, _, _ = model(inputs, relations)
    print(hidden)
