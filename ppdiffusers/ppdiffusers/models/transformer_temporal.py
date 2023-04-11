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

from dataclasses import dataclass
from typing import Optional

import paddle
import paddle.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .attention import BasicTransformerBlock
from .modeling_utils import ModelMixin


@dataclass
class TransformerTemporalModelOutput(BaseOutput):
    """
    Args:
        sample (`paddle.Tensor` of shape `(batch_size x num_frames, num_channels, height, width)`)
            Hidden states conditioned on `encoder_hidden_states` input.
    """

    sample: paddle.Tensor


class TransformerTemporalModel(ModelMixin, ConfigMixin):
    """
    Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
        double_self_attention (`bool`, *optional*):
            Configure if each TransformerBlock should contain two self-attention layers
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, epsilon=1e-06)
        self.proj_in = nn.Linear(in_features=in_channels, out_features=inner_dim)
        self.transformer_blocks = nn.LayerList(
            sublayers=[
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(in_features=inner_dim, out_features=in_channels)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        num_frames=1,
        cross_attention_kwargs=None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `paddle.Tensor` of shape `(batch size, num latent pixels)`.
                When continous, `paddle.Tensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `paddleTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `paddle.int64`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `paddle.Tensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.TransformerTemporalModelOutput`] or `tuple`:
            [`~models.transformer_2d.TransformerTemporalModelOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames
        residual = hidden_states
        hidden_states = hidden_states[(None), :].reshape((batch_size, num_frames, channel, height, width))
        hidden_states = hidden_states.transpose(perm=[0, 2, 1, 3, 4])
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(perm=[0, 3, 4, 2, 1]).reshape(
            (batch_size * height * width, num_frames, channel)
        )
        hidden_states = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states[(None), (None), :]
            .reshape((batch_size, height, width, channel, num_frames))
            .transpose(perm=[0, 3, 4, 1, 2])
        )
        hidden_states = hidden_states.reshape((batch_frames, channel, height, width))
        output = hidden_states + residual
        if not return_dict:
            return (output,)
        return TransformerTemporalModelOutput(sample=output)
