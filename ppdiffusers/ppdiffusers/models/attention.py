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
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import nn


class AttentionBlock(nn.Layer):
    """
    An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
    to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention.

    Parameters:
        channels (:obj:`int`): The number of channels in the input and output.
        num_head_channels (:obj:`int`, *optional*):
            The number of channels in each head. If None, then `num_heads` = 1.
        num_groups (:obj:`int`, *optional*, defaults to 32): The number of groups to use for group norm.
        rescale_output_factor (:obj:`float`, *optional*, defaults to 1.0): The factor to rescale the output by.
        eps (:obj:`float`, *optional*, defaults to 1e-5): The epsilon value to use for group norm.
    """

    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_channels=channels,
                                       num_groups=num_groups,
                                       epsilon=eps)

        # define q,k,v as linear layers
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels)

    def transpose_for_scores(self, projection: paddle.Tensor) -> paddle.Tensor:
        new_projection_shape = projection.shape[:-1] + [self.num_heads, -1]
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.reshape(new_projection_shape).transpose(
            [0, 2, 1, 3])
        return new_projection

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.reshape([batch, channel, height * width
                                               ]).transpose([0, 2, 1])

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        # transpose
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        # get scores
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attention_scores = paddle.matmul(query_states * scale,
                                         key_states * scale,
                                         transpose_y=True)  # TODO: use baddmm
        attention_probs = F.softmax(attention_scores.astype("float32"),
                                    axis=-1).astype(attention_scores.dtype)

        # compute attention output
        hidden_states = paddle.matmul(attention_probs, value_states)

        hidden_states = hidden_states.transpose([0, 2, 1, 3])
        new_hidden_states_shape = hidden_states.shape[:-2] + [
            self.channels,
        ]
        hidden_states = hidden_states.reshape(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose([0, 2, 1]).reshape(
            [batch, channel, height, width])

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class SpatialTransformer(nn.Layer):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Parameters:
        in_channels (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        depth (:obj:`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (:obj:`float`, *optional*, defaults to 0.1): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The number of context dimensions to use.
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        num_groups: int = 32,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=num_groups,
                                 num_channels=in_channels,
                                 epsilon=1e-6)

        self.proj_in = nn.Conv2D(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.LayerList([
            BasicTransformerBlock(inner_dim,
                                  n_heads,
                                  d_head,
                                  dropout=dropout,
                                  context_dim=context_dim) for d in range(depth)
        ])

        self.proj_out = nn.Conv2D(inner_dim,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def _set_attention_slice(self, slice_size):
        for block in self.transformer_blocks:
            block._set_attention_slice(slice_size)

    def forward(self, hidden_states, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.transpose([0, 2, 3, 1]).reshape(
            [batch, height * weight, inner_dim])
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context)
        hidden_states = hidden_states.reshape(
            [batch, height, weight, inner_dim]).transpose([0, 3, 1, 2])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states + residual


class BasicTransformerBlock(nn.Layer):
    r"""
    A basic Transformer block.

    Parameters:
        dim (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The size of the context vector for cross attention.
        gated_ff (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use a gated feed-forward network.
        checkpoint (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use checkpointing.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout=0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim,
                                    heads=n_heads,
                                    dim_head=d_head,
                                    dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def forward(self, hidden_states, context=None):
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states),
                                   context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class CrossAttention(nn.Layer):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (:obj:`int`): The number of channels in the query.
        context_dim (:obj:`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (:obj:`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (:obj:`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(self,
                 query_dim: int,
                 context_dim: Optional[int] = None,
                 heads: int = 8,
                 dim_head: int = 64,
                 dropout: int = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias_attr=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias_attr=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias_attr=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(
            [batch_size, seq_len, head_size, dim // head_size])
        tensor = tensor.transpose([0, 2, 1, 3]).reshape(
            [batch_size * head_size, seq_len, dim // head_size])
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(
            [batch_size // head_size, head_size, seq_len, dim])
        tensor = tensor.transpose([0, 2, 1, 3]).reshape(
            [batch_size // head_size, seq_len, dim * head_size])
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of

        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value,
                                                   sequence_length, dim)

        return self.to_out(hidden_states)

    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = paddle.matmul(query, key,
                                         transpose_y=True) * self.scale
        attention_probs = F.softmax(attention_scores, axis=-1)
        # compute attention output
        hidden_states = paddle.matmul(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = paddle.zeros(
            (batch_size_attention, sequence_length, dim // self.heads),
            dtype=query.dtype)
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[
            0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (paddle.matmul(query[start_idx:end_idx],
                                        key[start_idx:end_idx],
                                        transpose_y=True) * self.scale
                          )  # TODO: use baddbmm for better performance
            attn_slice = F.softmax(attn_slice, axis=-1)
            attn_slice = paddle.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class FeedForward(nn.Layer):
    r"""
    A feed-forward layer.

    Parameters:
        dim (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (:obj:`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        glu (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use GLU activation.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(self,
                 dim: int,
                 dim_out: Optional[int] = None,
                 mult: int = 4,
                 glu: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        return self.net(hidden_states)


# feedforward
class GEGLU(nn.Layer):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, axis=-1)
        return hidden_states * F.gelu(gate)
