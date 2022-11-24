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

from inspect import isfunction
from math import pi, log
from einops import rearrange, repeat

import paddle
from paddle import nn, einsum


# helper functions
def exists(val):
    return val is not None


# rotary embedding helper functions
def rotate_half(x):
    x = paddle.to_tensor(rearrange(x.numpy(), '... (d r) -> ... d r', r=2))
    x1, x2 = x.unbind(axis=-1)
    x = paddle.stack((-x2, x1), axis=-1)

    return paddle.to_tensor(rearrange(x.numpy(), '... d r -> ... (d r)'))


def apply_rotary_emb(freqs, t, start_index=0):
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[
        -1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[
        ..., end_index:]
    t = (t * paddle.cos(freqs)) + (rotate_half(t) * paddle.sin(freqs))
    return paddle.concat((t_left, t, t_right), axis=-1)


# learned rotation helpers
def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = paddle.to_tensor(
            rearrange(rotations.numpy(), '... r f -> ... (r f)'))
    rotations = paddle.repeat_interleave(rotations, repeats=2, axis=-1)
    return apply_rotary_emb(rotations, t, start_index=start_index)


# classes
class RotaryEmbedding(nn.Layer):

    def __init__(self,
                 dim,
                 custom_freqs=None,
                 freqs_for='lang',
                 theta=10000,
                 max_freq=10,
                 num_freqs=1,
                 learned_freq=False):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta**(paddle.cast(
                paddle.arange(0, dim, 2)[:(dim // 2)], paddle.float32) / dim))
        elif freqs_for == 'pixel':
            freqs = paddle.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = paddle.cast(paddle.ones(num_freqs), paddle.float32)
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()

        if learned_freq:
            self.gamma = paddle.create_parameter(
                shape=freqs.shape,
                dtype=str(freqs.numpy().dtype),
                default_initializer=paddle.nn.initializer.Assign(freqs))
        else:
            self.register_buffer('freqs', freqs)

    def rotate_queries_or_keys(self, t, seq_dim=-2):
        seq_len = t.shape[seq_dim]
        freqs = self.forward(lambda: paddle.arange(seq_len), cache_key=seq_len)
        return apply_rotary_emb(freqs, t)

    def forward(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs
        freqs = einsum('..., f -> ... f', paddle.cast(t, freqs.dtype), freqs)
        freqs = paddle.repeat_interleave(freqs,
                                         repeats=2,
                                         axis=len(freqs.shape) - 1)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs
