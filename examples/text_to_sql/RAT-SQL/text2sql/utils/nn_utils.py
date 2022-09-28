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
import json

import numpy as np
import paddle
from paddle import nn


def build_linear(n_in, n_out, name=None, init=None):
    return nn.Linear(
        n_in,
        n_out,
        weight_attr=paddle.ParamAttr(name='%s.w_0' %
                                     name if name is not None else None,
                                     initializer=init),
        bias_attr='%s.b_0' % name if name is not None else None,
    )


def build_layer_norm(n_in, name):
    return nn.LayerNorm(
        normalized_shape=n_in,
        weight_attr=paddle.ParamAttr(name='%s_layer_norm_scale' %
                                     name if name is not None else None,
                                     initializer=nn.initializer.Constant(1.)),
        bias_attr=paddle.ParamAttr(name='%s_layer_norm_bias' %
                                   name if name is not None else None,
                                   initializer=nn.initializer.Constant(0.)),
    )


def lstm_init(num_layers, hidden_size, *batch_sizes):
    init_size = batch_sizes + (hidden_size, )
    if num_layers is not None:
        init_size = (num_layers, ) + init_size
    init = paddle.zeros(init_size)
    return (init, init)


def batch_gather_2d(var, indices):
    """Gather slices from var in each batch, according to corrensponding
    index in indices. Currently, it only support 2d Tensor.

    Args:
        var (Variable): with shape [batch_size, ...]
        indices (Variable): with shape [batch_size, max_len]

    Returns: Variable with shape [batch_size]

    Raises: NULL

    Examples:
        var
            [[1, 2, 3],
             [4, 5, 6]]
        indices
            [[2, 0], [1, 2]]

        return
            [[3, 1], [5, 6]]

    """
    if len(indices.shape) != 2:
        raise ValueError('shape of indices error. it should be a 2-D layers. '
                         'but got shape = %s' % (str(indices.shape), ))

    batch_size = paddle.shape(indices)[0]

    zero = paddle.to_tensor([0], dtype='int64')
    one = paddle.to_tensor([1], dtype='int64')
    end = paddle.cast(batch_size, dtype='int64')
    batch_indices_1d = paddle.unsqueeze(
        paddle.arange(zero, end, one, dtype=indices.dtype), [1])

    seq_len = indices.shape[1]
    batch_indices = paddle.expand(batch_indices_1d, [batch_size, seq_len])

    coord_2d = paddle.concat(
        [paddle.unsqueeze(batch_indices, [2]),
         paddle.unsqueeze(indices, [2])],
        axis=2)
    coord_2d.stop_gradient = True
    coord_1d = paddle.reshape(coord_2d, shape=[-1, 2])
    output_1d = paddle.gather_nd(var, coord_1d)
    output_2d = paddle.reshape(output_1d, [batch_size, seq_len, var.shape[-1]])
    return output_2d


def sequence_mask(seq_hidden, mask, mode='zero'):
    """

    Args:
        seq_hidden (Tensor): NULL
        mask (Tensor): 1 for un-mask tokens, and 0 for mask tokens.
        mode (str): zero/-inf/+inf

    Returns: TODO

    Raises: NULL
    """
    dtype = seq_hidden.dtype

    while len(mask.shape) < len(seq_hidden.shape):
        mask = mask.unsqueeze([-1])

    mask = mask.cast(dtype=seq_hidden.dtype)
    masked = paddle.multiply(seq_hidden, mask)
    if mode == 'zero':
        return masked

    if mode == '-inf':
        scale_size = +1e5
    elif mode == '+inf':
        scale_size = -1e5
    else:
        raise ValueError(
            f'mask mode setting error. expect zero/-inf/+inf, but got {mode}')

    add_mask = paddle.scale(mask - 1, scale=scale_size)
    masked = paddle.add(masked, add_mask)
    return masked


def pad_sequences(seqs, max_len, value=0., dtype=np.int64):
    """padding sequences"""
    data_max_len = 0
    format_seqs = []
    for seq in seqs:
        format_seqs.append(list(seq))
        data_max_len = len(seq) if len(seq) > data_max_len else data_max_len
    max_len = min(max_len, data_max_len)
    padded = []
    for seq in format_seqs:
        padded.append(seq[:max_len] + [value] * (max_len - len(seq)))
    padded = np.array(padded)
    return padded.astype(dtype)


def pad_sequences_for_3d(seqs, max_col, max_num, dtype=np.int64):
    """padding sequences for 3d"""
    padded = []
    for seq in seqs:
        padded.append(
            np.vstack(
                (seq, np.zeros((max_col - seq.shape[0], max_num),
                               dtype=np.int64))))
    return np.array(padded).astype(dtype)


def pad_index_sequences(seqs, max_col, max_row, dtype=np.int64):
    """padding squences for column token indexs """
    padded = []
    for query in seqs:
        new_cols = []
        for col in query[:max_row]:
            temp_cols = col[:max_col] + [0] * (max_col - len(col))
            new_cols.append(temp_cols)
        new_cols = new_cols + [[0] * max_col
                               for _ in range(max_row - len(new_cols))]
        padded.append(new_cols)
    return np.array(padded).astype(dtype)


def tensor2numpy(inputs):
    if type(inputs) in (list, tuple):
        return [x.numpy() for x in inputs]
    elif type(inputs) is dict:
        outputs = {}
        for key, value in inputs.items():
            if type(value) is paddle.Tensor:
                outputs[key] = value.numpy()
            else:
                outputs[key] = value
        return outputs
    elif type(inputs) is paddle.Tensor:
        return inputs.numpy()
    else:
        raise ValueError('only support inputs to be of type list/tuple/dict/Tensor.' + \
                         f'but got {type(inputs)}')


if __name__ == "__main__":
    """run some simple test cases"""
    seq_input = paddle.to_tensor([
        [1, 2, 3, 4],
        [5, 5, 5, 5],
    ],
                                 dtype='float32')
    mask = paddle.to_tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
    ], dtype='float32')

    print(sequence_mask(seq_input, mask, mode='zero'))
    print(sequence_mask(seq_input, mask, mode='-inf'))
