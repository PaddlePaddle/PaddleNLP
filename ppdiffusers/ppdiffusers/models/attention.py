# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from ..utils import is_ppxformers_available
from .cross_attention import CrossAttention
from .embeddings import CombinedTimestepLabelEmbeddings


class AttentionBlock(nn.Layer):
    """
    An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
    to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention.

    Parameters:
        channels (`int`): The number of channels in the input and output.
        num_head_channels (`int`, *optional*):
            The number of channels in each head. If None, then `num_heads` = 1.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for group norm.
        rescale_output_factor (`float`, *optional*, defaults to 1.0): The factor to rescale the output by.
        eps (`float`, *optional*, defaults to 1e-5): The epsilon value to use for group norm.
    """

    # IMPORTANT;TODO(Patrick, William) - this class will be deprecated soon. Do not use it anymore

    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.head_size = self.head_size = self.channels // self.num_heads
        self.scale = 1 / math.sqrt(self.channels / self.num_heads)

        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, epsilon=eps)

        # define q,k,v as linear layers
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels)

        self._use_memory_efficient_attention_xformers = False
        self._attention_op = None

    def reshape_heads_to_batch_dim(self, tensor, transpose=True):
        tensor = tensor.reshape([0, 0, self.num_heads, self.head_size])
        if transpose:
            tensor = tensor.transpose([0, 2, 1, 3])
        return tensor

    def reshape_batch_dim_to_heads(self, tensor, transpose=True):
        if transpose:
            tensor = tensor.transpose([0, 2, 1, 3])
        tensor = tensor.reshape([0, 0, tensor.shape[2] * tensor.shape[3]])
        return tensor

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[str] = None
    ):
        if self.head_size > 128 and attention_op == "flash":
            attention_op = "cutlass"
        if use_memory_efficient_attention_xformers:
            if not is_ppxformers_available():
                raise NotImplementedError(
                    "requires the scaled_dot_product_attention but your PaddlePaddle donot have this. Checkout the instructions on the installation page: https://www.paddlepaddle.org.cn/install/quick and follow the ones that match your environment."
                )
            else:
                try:
                    _ = F.scaled_dot_product_attention_(
                        paddle.randn((1, 1, 2, 40), dtype=paddle.float16),
                        paddle.randn((1, 1, 2, 40), dtype=paddle.float16),
                        paddle.randn((1, 1, 2, 40), dtype=paddle.float16),
                        attention_op=attention_op,
                    )
                except Exception as e:
                    raise e

        self._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        self._attention_op = attention_op

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.reshape([batch, channel, height * width]).transpose([0, 2, 1])

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_proj = self.reshape_heads_to_batch_dim(
            query_proj, transpose=not self._use_memory_efficient_attention_xformers
        )
        key_proj = self.reshape_heads_to_batch_dim(
            key_proj, transpose=not self._use_memory_efficient_attention_xformers
        )
        value_proj = self.reshape_heads_to_batch_dim(
            value_proj, transpose=not self._use_memory_efficient_attention_xformers
        )

        if self._use_memory_efficient_attention_xformers:
            hidden_states = F.scaled_dot_product_attention_(
                query_proj,
                key_proj,
                value_proj,
                attn_mask=None,
                scale=self.scale,
                dropout_p=0.0,
                training=self.training,
                attention_op=self._attention_op,
            )
        else:
            attention_scores = paddle.matmul(query_proj, key_proj, transpose_y=True) * self.scale
            attention_probs = F.softmax(attention_scores.cast("float32"), axis=-1).cast(attention_scores.dtype)
            hidden_states = paddle.matmul(attention_probs, value_proj)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(
            hidden_states, transpose=not self._use_memory_efficient_attention_xformers
        )

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 1]).reshape([batch, channel, height, width])

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class BasicTransformerBlock(nn.Layer):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # 1. Self-Attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.attn2 = None

        if not norm_elementwise_affine:
            norm_kwargs = {"weight_attr": False, "bias_attr": False}
        else:
            norm_kwargs = {}

        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, **norm_kwargs)

        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim, **norm_kwargs)
            )
        else:
            self.norm2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, **norm_kwargs)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            # 2. Cross-Attention
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class FeedForward(nn.Layer):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.LayerList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GELU(nn.Layer):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate
        self.approximate_bool = approximate == "tanh"

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=self.approximate_bool)
        return hidden_states


class GEGLU(nn.Layer):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, axis=-1)
        return hidden_states * F.gelu(gate)


class ApproximateGELU(nn.Layer):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * F.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Layer):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.Silu()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        # elementwise_affine=False
        norm_kwargs = {"weight_attr": False, "bias_attr": False}
        self.norm = nn.LayerNorm(embedding_dim, **norm_kwargs)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        # must set axis=-1, paddle vs pytorch
        scale, shift = paddle.chunk(emb, 2, axis=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Layer):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()

        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)

        self.silu = nn.Silu()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias_attr=True)
        # elementwise_affine=False
        norm_kwargs = {"weight_attr": False, "bias_attr": False}
        self.norm = nn.LayerNorm(embedding_dim, epsilon=1e-6, **norm_kwargs)

    def forward(self, x, timestep, class_labels, hidden_dtype=None):
        emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaGroupNorm(nn.Layer):
    """
    GroupNorm layer modified to incorporate timestep embeddings.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.act = None
        if act_fn == "swish":
            self.act = lambda x: F.silu(x)
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "silu":
            self.act = nn.Silu()
        elif act_fn == "gelu":
            self.act = nn.GELU()

        self.linear = nn.Linear(embedding_dim, out_dim * 2)
        # elementwise_affine=False
        norm_kwargs = {"weight_attr": False, "bias_attr": False}
        self.group_norm = nn.GroupNorm(num_groups, out_dim, epsilon=eps, **norm_kwargs)
        self.group_norm.weight = None
        self.group_norm.bias = None

    def forward(self, x, emb):
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, axis=1)
        x = self.group_norm(x)
        x = x * (1 + scale) + shift
        return x
