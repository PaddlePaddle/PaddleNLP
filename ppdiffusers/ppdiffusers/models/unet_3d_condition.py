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
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from .embeddings import TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .transformer_temporal import TransformerTemporalModel
from .unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)

logger = logging.get_logger(__name__)


@dataclass
class UNet3DConditionOutput(BaseOutput):
    """
    Args:
        sample (`paddle.Tensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: paddle.Tensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    """
    UNet3DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, it will skip the normalization and activation layers in post-processing
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str] = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-05,
        cross_attention_dim: int = 1024,
        attention_head_dim: Union[int, Tuple[int]] = 64,
    ):
        super().__init__()
        self.sample_size = sample_size
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )
        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )
        conv_in_kernel = 3
        conv_out_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2D(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=conv_in_padding,
        )
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        self.transformer_in = TransformerTemporalModel(
            num_attention_heads=8,
            attention_head_dim=attention_head_dim,
            in_channels=block_out_channels[0],
            num_layers=1,
        )
        self.down_blocks = nn.LayerList(sublayers=[])
        self.up_blocks = nn.LayerList(sublayers=[])
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=False,
            )
            self.down_blocks.append(down_block)
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
        )
        self.num_upsamplers = 0
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=False,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=block_out_channels[0],
                epsilon=norm_eps,
                weight_attr=None,
                bias_attr=None,
            )
            self.conv_act = nn.Silu()
        else:
            self.conv_norm_out = None
            self.conv_act = None
        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2D(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=conv_out_kernel,
            padding=conv_out_padding,
        )

    def set_attention_slice(self, slice_size):
        """
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: nn.Layer):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)
            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)
        num_slicable_layers = len(sliceable_head_dims)
        if slice_size == "auto":
            slice_size = [(dim // 2) for dim in sliceable_head_dims]
        elif slice_size == "max":
            slice_size = num_slicable_layers * [1]
        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size
        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )
        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        def fn_recursive_set_attention_slice(module: nn.Layer, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())
            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: paddle.Tensor,
        timestep: Union[paddle.Tensor, float, int],
        encoder_hidden_states: paddle.Tensor,
        class_labels: Optional[paddle.Tensor] = None,
        timestep_cond: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[paddle.Tensor]] = None,
        mid_block_additional_residual: Optional[paddle.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        """
        Args:
            sample (`paddle.Tensor`): (batch, num_frames, channel, height, width) noisy inputs tensor
            timestep (`paddle.Tensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`paddle.Tensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet3DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in ppdiffusers.cross_attention.

        Returns:
            [`~models.unet_2d_condition.UNet3DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet3DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # TODO junnyu, add this to support pure fp16
        sample = sample.cast(self.dtype)
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.astype(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(axis=1)
        timesteps = timestep
        if not paddle.is_tensor(x=timesteps):
            if isinstance(timestep, float):
                dtype = "float64"
            else:
                dtype = "int64"
            timesteps = paddle.to_tensor([timesteps], dtype=dtype)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        num_frames = sample.shape[2]
        timesteps = timesteps.expand(
            [
                sample.shape[0],
            ]
        )
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.astype(dtype=self.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        emb = emb.repeat_interleave(repeats=num_frames, axis=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, axis=0)
        sample = sample.transpose(perm=[0, 2, 1, 3, 4]).reshape(
            (sample.shape[0] * num_frames, -1) + tuple(sample.shape[3:])
        )
        sample = self.conv_in(sample)
        sample = self.transformer_in(sample, num_frames=num_frames).sample
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)
            down_block_res_samples += res_samples
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        sample = sample[(None), :].reshape((-1, num_frames) + tuple(sample.shape[1:])).transpose(perm=[0, 2, 1, 3, 4])
        if not return_dict:
            return (sample,)
        return UNet3DConditionOutput(sample=sample)
