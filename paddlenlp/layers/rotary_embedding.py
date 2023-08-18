# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len_cached = max_position_embeddings

        dtype = paddle.get_default_dtype()
        inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, dim, 2), dtype="float32") / dim))  # [dim / 2]
        self.register_buffer("inv_freq", inv_freq.cast(dtype))

        # higher acc using float32
        t = paddle.arange(max_position_embeddings, dtype="float32")  # [max_position_embeddings]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq.cast("float32"))  # [max_position_embeddings, dim/2]
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat([freqs, freqs], axis=-1)  # [max_position_embeddings, dim]
        # [bs, seqlen, nhead, head_dim]
        self.cos_cached = emb.cos()[None, :, None, :]  # [1, max_pos, 1, dim]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:, :seq_len, :, ...],
            self.sin_cached[:, :seq_len, :, ...],
        )

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        if len(position_ids.shape) != 2:
            raise ValueError("position_ids should be a 2D tensor")
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        q_embed = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)
        return q_embed, k_embed
