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

__all__ = [
    "RotaryEmbedding",
    "LinearScalingRotaryEmbedding",
    "NTKScalingRotaryEmbedding",
    "DynamicNTKScalingRotaryEmbedding",
]


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = paddle.get_default_dtype()
        # [dim / 2]
        self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :].cast(self.dtype)
        self.sin_cached = emb.sin()[None, :, None, :].cast(self.dtype)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].cast(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].cast(x.dtype),
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


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        t = t / self.scaling_factor
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :].cast(self.dtype)
        self.sin_cached = emb.sin()[None, :, None, :].cast(self.dtype)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].cast(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].cast(x.dtype),
        )


class NTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with NTK scaling. https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        base = base * scaling_factor ** (dim / (dim - 2))
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _scale_cos_sin(self, seq_len):
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        alpha = (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
        base = self.base * alpha ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))
        freqs = paddle.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        scale_cos = emb.cos()[None, :, None, :].cast(self.dtype)
        scale_sin = emb.sin()[None, :, None, :].cast(self.dtype)
        return scale_cos, scale_sin

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_position_embeddings:
            scale_cos, scale_sin = self._scale_cos_sin(seq_len=seq_len)
            return (
                scale_cos[:, :, :seq_len, ...].cast(x.dtype),
                scale_sin[:, :, :seq_len, ...].cast(x.dtype),
            )
        else:
            return (
                self.cos_cached[:, :, :seq_len, ...].cast(x.dtype),
                self.sin_cached[:, :, :seq_len, ...].cast(x.dtype),
            )
