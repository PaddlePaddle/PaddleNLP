# coding:utf-8

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return paddle.cast(paddle.ones(shape), dtype=paddle.bool)
    elif prob == 0:
        return paddle.cast(paddle.zeros(shape), dtype=paddle.bool)
    else:
        return paddle.cast(paddle.zeros(shape), paddle.float32).uniform_(
            0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


def Downsample(dim):
    return nn.Conv3D(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Upsample(dim):
    return nn.Conv3DTranspose(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class EinopsToAndFrom(nn.Layer):

    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops, shape)))
        reconstitute_indices = dict(
            tuple(zip(self.from_einops, list(range(len(shape))))))
        x = paddle.transpose(
            x, [reconstitute_indices[a] for a in list(''.join(self.to_einops))])
        x = paddle.reshape(x, [
            reconstitute_kwargs[a[0]] *
            reconstitute_kwargs[a[1]] if len(a) == 2 else reconstitute_kwargs[a]
            for a in self.to_einops
        ])

        x = self.fn(x, **kwargs)
        reconstitute_shape = []
        for a in list(''.join(self.to_einops)):
            if len(a) == 2:
                reconstitute_shape.extend(
                    [reconstitute_kwargs[a[0]], reconstitute_kwargs[a[1]]])
            else:
                reconstitute_shape.extend([reconstitute_kwargs[a]])
        x = paddle.reshape(x, reconstitute_shape)
        reconstitute_indices = dict(
            tuple(zip(list(''.join(self.to_einops)), list(range(len(shape))))))
        x = paddle.transpose(
            x,
            [reconstitute_indices[a] for a in list(''.join(self.from_einops))])

        return x
