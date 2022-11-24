# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import math
import numpy as np

import paddle
from paddle import nn


def get_timestep_embedding(
    timesteps: paddle.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * paddle.arange(
        start=0, end=half_dim, dtype="float32")
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = paddle.exp(exponent)
    emb = timesteps[:, None].astype("float32") * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = paddle.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = paddle.concat(emb, paddle.zeros([emb.shape[0], 1]), axis=-1)
    return emb


class TimestepEmbedding(nn.Layer):

    def __init__(self, channel: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()

        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.act = None
        if act_fn == "silu":
            self.act = nn.Silu()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Layer):

    def __init__(self, num_channels: int, flip_sin_to_cos: bool,
                 downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class GaussianFourierProjection(nn.Layer):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.register_buffer("weight", paddle.randn(embedding_size) * scale)

        # to delete later
        self.register_buffer("W", paddle.randn(embedding_size) * scale)

        self.weight = self.W

    def forward(self, x):
        x = paddle.log(x)
        x_proj = x[:, None] * self.weight[None, :] * 2 * np.pi
        out = paddle.concat([paddle.sin(x_proj), paddle.cos(x_proj)], axis=-1)
        return out


class RelativePositionBias(nn.Layer):

    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  num_buckets=32,
                                  max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += paddle.cast((n < 0), dtype=paddle.int64) * num_buckets
        n = paddle.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + paddle.cast(
            (paddle.log(paddle.cast(n, dtype=paddle.float32) / max_exact) /
             math.log(max_distance / max_exact) * (num_buckets - max_exact)),
            dtype=paddle.int64)
        sub = paddle.full_like(val_if_large, num_buckets - 1)
        val_if_large = paddle.minimum(val_if_large, sub)

        ret += paddle.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device=None):
        q_pos = paddle.arange(n, dtype=paddle.int64)
        k_pos = paddle.arange(n, dtype=paddle.int64)
        rel_pos = paddle.reshape(k_pos, [1, k_pos.shape[0]]) - paddle.reshape(
            q_pos, [q_pos[0], 1])
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return paddle.transpose(values, [2, 0, 1])


class SinusoidalPosEmb(nn.Layer):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(paddle.arange(half_dim) * -emb)
        emb = paddle.cast(x[:, None], dtype=paddle.float32) * emb[None, :]
        emb = paddle.concat((paddle.sin(paddle.cast(emb, paddle.float32)),
                             paddle.cos(paddle.cast(emb, paddle.float32))),
                            axis=-1)
        return emb
