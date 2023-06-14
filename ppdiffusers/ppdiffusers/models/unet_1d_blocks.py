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
from .resnet import Downsample1D, ResidualTemporalBlock1D, Upsample1D, rearrange_dims


class DownResnetBlock1D(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_layers=1,
        conv_shortcut=False,
        temb_channels=32,
        groups=32,
        groups_out=None,
        non_linearity=None,
        time_embedding_norm="default",
        output_scale_factor=1.0,
        add_downsample=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.add_downsample = add_downsample
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        self.resnets = nn.LayerList(resnets)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.Silu()
        else:
            self.nonlinearity = None

        self.downsample = None
        if add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True, padding=1)

    def forward(self, hidden_states, temb=None):
        output_states = ()

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        output_states += (hidden_states,)

        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states, output_states


class UpResnetBlock1D(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_layers=1,
        temb_channels=32,
        groups=32,
        groups_out=None,
        non_linearity=None,
        time_embedding_norm="default",
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm
        self.add_upsample = add_upsample
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(2 * in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        self.resnets = nn.LayerList(resnets)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.Silu()
        else:
            self.nonlinearity = None

        self.upsample = None
        if add_upsample:
            self.upsample = Upsample1D(out_channels, use_conv_transpose=True)

    def forward(self, hidden_states, res_hidden_states_tuple=None, temb=None):
        if res_hidden_states_tuple is not None:
            res_hidden_states = res_hidden_states_tuple[-1]
            hidden_states = paddle.concat((hidden_states, res_hidden_states), axis=1)

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class ValueFunctionMidBlock1D(nn.Layer):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.res1 = ResidualTemporalBlock1D(in_channels, in_channels // 2, embed_dim=embed_dim)
        self.down1 = Downsample1D(out_channels // 2, use_conv=True)
        self.res2 = ResidualTemporalBlock1D(in_channels // 2, in_channels // 4, embed_dim=embed_dim)
        self.down2 = Downsample1D(out_channels // 4, use_conv=True)

    def forward(self, x, temb=None):
        x = self.res1(x, temb)
        x = self.down1(x)
        x = self.res2(x, temb)
        x = self.down2(x)
        return x


class MidResTemporalBlock1D(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        num_layers: int = 1,
        add_downsample: bool = False,
        add_upsample: bool = False,
        non_linearity=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_downsample = add_downsample

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=embed_dim)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=embed_dim))

        self.resnets = nn.LayerList(resnets)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.Silu()
        else:
            self.nonlinearity = None

        self.upsample = None
        if add_upsample:
            self.upsample = Downsample1D(out_channels, use_conv=True)

        self.downsample = None
        if add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True)

        if self.upsample and self.downsample:
            raise ValueError("Block cannot downsample and upsample")

    def forward(self, hidden_states, temb):
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.upsample:
            hidden_states = self.upsample(hidden_states)
        if self.downsample:
            self.downsample = self.downsample(hidden_states)

        return hidden_states


class OutConv1DBlock(nn.Layer):
    def __init__(self, num_groups_out, out_channels, embed_dim, act_fn):
        super().__init__()
        self.final_conv1d_1 = nn.Conv1D(embed_dim, embed_dim, 5, padding=2)
        self.final_conv1d_gn = nn.GroupNorm(num_groups_out, embed_dim)
        if act_fn == "silu":
            self.final_conv1d_act = nn.Silu()
        if act_fn == "mish":
            self.final_conv1d_act = nn.Mish()
        self.final_conv1d_2 = nn.Conv1D(embed_dim, out_channels, 1)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.final_conv1d_1(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_gn(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_act(hidden_states)
        hidden_states = self.final_conv1d_2(hidden_states)
        return hidden_states


class OutValueFunctionBlock(nn.Layer):
    def __init__(self, fc_dim, embed_dim):
        super().__init__()
        self.final_block = nn.LayerList(
            [
                nn.Linear(fc_dim + embed_dim, fc_dim // 2),
                nn.Mish(),
                nn.Linear(fc_dim // 2, 1),
            ]
        )

    def forward(self, hidden_states, temb):
        hidden_states = hidden_states.reshape([hidden_states.shape[0], -1])
        hidden_states = paddle.concat((hidden_states, temb), axis=-1)
        for layer in self.final_block:
            hidden_states = layer(hidden_states)

        return hidden_states


_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}


class Downsample1d(nn.Layer):
    def __init__(self, kernel="linear", pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = paddle.to_tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, hidden_states):
        hidden_states = F.pad(hidden_states, (self.pad,) * 2, self.pad_mode, data_format="NCL")
        weight = paddle.zeros(
            [hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]], dtype=hidden_states.dtype
        )
        indices = paddle.arange(hidden_states.shape[1])
        weight[indices, indices] = self.kernel.cast(weight.dtype)
        return F.conv1d(hidden_states, weight, stride=2)


class Upsample1d(nn.Layer):
    def __init__(self, kernel="linear", pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = paddle.to_tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, hidden_states, temb=None):
        hidden_states = F.pad(hidden_states, ((self.pad + 1) // 2,) * 2, self.pad_mode, data_format="NCL")
        weight = paddle.zeros(
            [hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]], dtype=hidden_states.dtype
        )
        indices = paddle.arange(hidden_states.shape[1])
        weight[indices, indices] = self.kernel.cast(weight.dtype)
        return F.conv1d_transpose(hidden_states, weight, stride=2, padding=self.pad * 2 + 1)


class SelfAttention1d(nn.Layer):
    def __init__(self, in_channels, n_head=1, dropout_rate=0.0):
        super().__init__()
        self.channels = in_channels
        self.group_norm = nn.GroupNorm(1, num_channels=in_channels)
        self.num_heads = n_head
        self.head_size = in_channels // n_head
        self.scale = 1 / math.sqrt(self.head_size)

        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels, self.channels)
        self.value = nn.Linear(self.channels, self.channels)

        self.proj_attn = nn.Linear(self.channels, self.channels)

        self.dropout = nn.Dropout(dropout_rate)

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

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.transpose([0, 2, 1])

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
        hidden_states = hidden_states.transpose([0, 2, 1])
        hidden_states = self.dropout(hidden_states)

        output = hidden_states + residual

        return output


class ResConvBlock(nn.Layer):
    def __init__(self, in_channels, mid_channels, out_channels, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.has_conv_skip = in_channels != out_channels

        if self.has_conv_skip:
            self.conv_skip = nn.Conv1D(in_channels, out_channels, 1, bias_attr=False)

        self.conv_1 = nn.Conv1D(in_channels, mid_channels, 5, padding=2)
        self.group_norm_1 = nn.GroupNorm(1, mid_channels)
        self.gelu_1 = nn.GELU()
        self.conv_2 = nn.Conv1D(mid_channels, out_channels, 5, padding=2)

        if not self.is_last:
            self.group_norm_2 = nn.GroupNorm(1, out_channels)
            self.gelu_2 = nn.GELU()

    def forward(self, hidden_states):
        residual = self.conv_skip(hidden_states) if self.has_conv_skip else hidden_states

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.group_norm_1(hidden_states)
        hidden_states = self.gelu_1(hidden_states)
        hidden_states = self.conv_2(hidden_states)

        if not self.is_last:
            hidden_states = self.group_norm_2(hidden_states)
            hidden_states = self.gelu_2(hidden_states)

        output = hidden_states + residual
        return output


class UNetMidBlock1D(nn.Layer):
    def __init__(self, mid_channels, in_channels, out_channels=None):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]
        self.up = Upsample1d(kernel="cubic")

        self.attentions = nn.LayerList(attentions)
        self.resnets = nn.LayerList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.down(hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class AttnDownBlock1D(nn.Layer):
    def __init__(self, out_channels, in_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        self.attentions = nn.LayerList(attentions)
        self.resnets = nn.LayerList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.down(hidden_states)

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        return hidden_states, (hidden_states,)


class DownBlock1D(nn.Layer):
    def __init__(self, out_channels, in_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.LayerList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.down(hidden_states)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states, (hidden_states,)


class DownBlock1DNoSkip(nn.Layer):
    def __init__(self, out_channels, in_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.LayerList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = paddle.concat([hidden_states, temb], axis=1)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states, (hidden_states,)


class AttnUpBlock1D(nn.Layer):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        self.attentions = nn.LayerList(attentions)
        self.resnets = nn.LayerList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = paddle.concat([hidden_states, res_hidden_states], axis=1)

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class UpBlock1D(nn.Layer):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.LayerList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = paddle.concat([hidden_states, res_hidden_states], axis=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class UpBlock1DNoSkip(nn.Layer):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, is_last=True),
        ]

        self.resnets = nn.LayerList(resnets)

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = paddle.concat([hidden_states, res_hidden_states], axis=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states


def get_down_block(down_block_type, num_layers, in_channels, out_channels, temb_channels, add_downsample):
    if down_block_type == "DownResnetBlock1D":
        return DownResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )
    elif down_block_type == "DownBlock1D":
        return DownBlock1D(out_channels=out_channels, in_channels=in_channels)
    elif down_block_type == "AttnDownBlock1D":
        return AttnDownBlock1D(out_channels=out_channels, in_channels=in_channels)
    elif down_block_type == "DownBlock1DNoSkip":
        return DownBlock1DNoSkip(out_channels=out_channels, in_channels=in_channels)
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(up_block_type, num_layers, in_channels, out_channels, temb_channels, add_upsample):
    if up_block_type == "UpResnetBlock1D":
        return UpResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
        )
    elif up_block_type == "UpBlock1D":
        return UpBlock1D(in_channels=in_channels, out_channels=out_channels)
    elif up_block_type == "AttnUpBlock1D":
        return AttnUpBlock1D(in_channels=in_channels, out_channels=out_channels)
    elif up_block_type == "UpBlock1DNoSkip":
        return UpBlock1DNoSkip(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"{up_block_type} does not exist.")


def get_mid_block(mid_block_type, num_layers, in_channels, mid_channels, out_channels, embed_dim, add_downsample):
    if mid_block_type == "MidResTemporalBlock1D":
        return MidResTemporalBlock1D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            add_downsample=add_downsample,
        )
    elif mid_block_type == "ValueFunctionMidBlock1D":
        return ValueFunctionMidBlock1D(in_channels=in_channels, out_channels=out_channels, embed_dim=embed_dim)
    elif mid_block_type == "UNetMidBlock1D":
        return UNetMidBlock1D(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels)
    raise ValueError(f"{mid_block_type} does not exist.")


def get_out_block(*, out_block_type, num_groups_out, embed_dim, out_channels, act_fn, fc_dim):
    if out_block_type == "OutConv1DBlock":
        return OutConv1DBlock(num_groups_out, out_channels, embed_dim, act_fn)
    elif out_block_type == "ValueFunction":
        return OutValueFunctionBlock(fc_dim, embed_dim)
    return None
