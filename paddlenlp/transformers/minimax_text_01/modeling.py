# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

from .configuration import MiniMaxText01Config

BLOCK = 256


def get_activation_fn(activation):
    print(f"activation: {activation}")
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":

        def f(x):
            with paddle.no_grad():
                x_max = paddle.max(x, axis=-1, keepdim=True)
            y = paddle.exp(x - x_max)
            return y

        return f
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":

        def f(x):
            return 1 + F.elu(x)

        return f
    elif activation == "2+elu":

        def f(x):
            return 2 + F.elu(x)

        return f
    elif activation == "silu" or activation == "swish":
        return F.silu
    elif activation == "sine":
        return paddle.sin
    else:
        return lambda x: x


class MiniMaxText01RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super(MiniMaxText01RMSNorm, self).__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size], dtype="float32", default_initializer=paddle.nn.initializer.Constant(1.0)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = paddle.cast(hidden_states, dtype="float32")

        variance = paddle.mean(paddle.square(hidden_states), axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)

        return self.weight * paddle.cast(hidden_states, dtype=input_dtype)


class MiniMaxText01LightningAttention(nn.Layer):
    def __init__(self, config: MiniMaxText01Config, layer_idx: Optional[int] = None):
        super().__init__()
        paddle.seed(42)  # add
        bias = False
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

        self.out_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias_attr=bias)
        self.act = get_activation_fn(config.hidden_act)
        self.norm = MiniMaxText01RMSNorm(self.head_dim * self.num_heads)

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.head_dim * self.num_heads, bias_attr=bias)

        self.output_gate = nn.Linear(self.hidden_size, self.head_dim * self.num_heads, bias_attr=bias)

        # for inference only
        self.offset = 0
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor] = None,  # (b, h, n, m)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[Tensor]] = None,
        use_cache: bool = False,
        slope_rate: Optional[Tensor] = None,
        **kwargs
    ):
        if not self.training:
            return self.inference(
                hidden_states,
                attn_mask,
                output_attentions,
                past_key_value,
                use_cache,
                slope_rate,
            )

    def inference(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,  # (b, n)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[Tensor]] = None,
        use_cache: bool = False,
        slope_rate: Optional[Tensor] = None,  # (h, 1, 1)
    ):
        b, n, d = x.shape
        qkv = self.act(self.qkv_proj(x))
        new_shape = list(qkv.shape[:-1]) + [self.num_heads, -1]
        qkv = qkv.reshape(new_shape)
        q, k, v = paddle.split(qkv, [self.head_dim] * 3, axis=-1)
        q = paddle.transpose(q, perm=[0, 2, 1, 3])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        v = paddle.transpose(v, perm=[0, 2, 1, 3])

        if past_key_value is None:
            self.offset = q.shape[-2]
        else:
            self.offset += 1

        ratio = paddle.exp(-slope_rate)

        if past_key_value is None:
            slope_rate = paddle.cast(slope_rate, dtype="float32")
            if attn_mask is not None:
                v = paddle.masked_fill(v, (1 - attn_mask).unsqueeze(1).unsqueeze(-1).astype(paddle.bool), 0)

            NUM_BLOCK = (n + BLOCK - 1) // BLOCK
            b, h, n, d = q.shape
            e = v.shape[-1]
            array = paddle.arange(BLOCK).to(q.place).astype("float32") + 1
            q_decay = paddle.exp(-slope_rate * array.reshape([-1, 1]))
            k_decay = paddle.exp(-slope_rate * (BLOCK - array.reshape([-1, 1])))
            index = array[:, None] - array[None, :]

            s_index = slope_rate * paddle.unsqueeze(paddle.unsqueeze(index, axis=0), axis=0)
            s_index = paddle.where(index >= 0, -s_index, float("-inf"))
            diag_decay = paddle.exp(s_index)

            kv = paddle.zeros([b, h, d, e], dtype="float32")
            kv = kv.to(q.place)
            output = paddle.empty([b, h, n, e], dtype=q.dtype)
            output = output.to(q.place)
            for i in range(NUM_BLOCK):
                si = i * BLOCK
                ei = min(si + BLOCK, n)
                m = ei - si
                qi = q[:, :, si:ei]
                ki = k[:, :, si:ei]
                vi = v[:, :, si:ei]

                qkv_none_diag = paddle.matmul(qi * q_decay[:, :m], kv).to("float32")

                qk = (
                    paddle.matmul(qi, paddle.transpose(ki, perm=[0, 1, 3, 2])).astype(paddle.float32)
                    * diag_decay[:, :, :m, :m]
                )

                qkv_diag = paddle.matmul(qk.astype(paddle.float32), vi.astype(paddle.float32))

                block_decay = paddle.exp(-slope_rate * m)
                output[:, :, si:ei] = qkv_none_diag + qkv_diag
                kv = block_decay * kv + paddle.matmul((ki * k_decay[:, -m:]).transpose([0, 1, 3, 2]), vi)

        else:
            kv = past_key_value
            output = []
            for i in range(n):
                kv = ratio * kv + paddle.einsum(
                    "... n d, ... n e -> ... d e",
                    k[:, :, i : i + 1],
                    v[:, :, i : i + 1],
                )
                qkv = paddle.einsum("... n e, ... e d -> ... n d", q[:, :, i : i + 1], kv)
                output.append(qkv)
            output = paddle.concat(output, axis=-2)

        output = output.reshape([b, n, h * d])
        print(output.shape, output[0][0])

        output = self.norm(output)

        output = F.sigmoid(self.output_gate(x)) * output

        output = self.out_proj(output)

        attn_weights = None

        return output, attn_weights, kv


class MiniMaxText01RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (paddle.arange(0, self.dim, 2, dtype="float32") / self.dim))
        self.inv_freq = paddle.to_tensor(inv_freq, stop_gradient=True)

        # Build here to make `paddle.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.place, dtype="float32")

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached, dtype="int64")
        t = t.astype("float32")

        freqs = paddle.outer(t, self.inv_freq)
        emb = paddle.concat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().astype(dtype)
        self.sin_cached = emb.sin().astype(dtype)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.place, dtype="float32")

        return (
            self.cos_cached[:seq_len].astype("float32"),
            self.sin_cached[:seq_len].astype("float32"),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    dtype = q.dtype
    rot_dim = cos.shape[-1]
    q_, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    # Unsqueeze the cosine and sine tensors
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    # Apply rotary embedding using the custom rotate_half function
    q_embed = (q_ * cos) + (rotate_half(q_) * sin)
    k_embed = (k_ * cos) + (rotate_half(k_) * sin)

    # Concatenate q_embed and k_embed with their respective pass tensors and convert to the correct dtype
    return paddle.concat((q_embed, q_pass), axis=-1).astype(dtype), paddle.concat((k_embed, k_pass), axis=-1).astype(
        dtype
    )


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(x, axis=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])
