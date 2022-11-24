# coding:utf-8

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

import os
import math
import time
from functools import partial
from dataclasses import dataclass

import paddle
from paddle import nn
import paddle.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ..utils import BaseOutput
from .embeddings import RelativePositionBias, SinusoidalPosEmb
from .attention import SpatialTemporalAttention, SpatialLinearAttention
from .rotary_embedding import RotaryEmbedding
from .resnet import Residual, PreNorm, ResnetBlock3D
from .video_utils import exists, default, is_odd, prob_mask_like, Downsample, Upsample, EinopsToAndFrom

BERT_MODEL_DIM = 768


@dataclass
class UNet3DOutput(BaseOutput):
    """
    Args:
        sample (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: paddle.Tensor


class UNet3DModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 dim,
                 cond_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 attn_heads=8,
                 attn_dim_head=32,
                 use_bert_text_cond=False,
                 init_dim=None,
                 init_kernel_size=7,
                 use_sparse_linear_attn=True,
                 block_type='resnet',
                 resnet_groups=8):
        super().__init__()
        self.channels = channels
        self.use_bert_text_cond = use_bert_text_cond

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom(
            # 'b c f h w', 'b (h w) f c',
            ['b', 'c', 'f', 'h', 'w'],
            ['b', 'hw', 'f', 'c'],
            SpatialTemporalAttention(dim,
                                     heads=attn_heads,
                                     dim_head=attn_dim_head,
                                     rotary_emb=rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32
        )  # realistically will not be able to generate that many frames of video... yet

        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3D(channels,
                                   init_dim,
                                   (1, init_kernel_size, init_kernel_size),
                                   padding=(0, init_padding, init_padding))
        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim),
                                      nn.Linear(dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))

        # text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        x = paddle.randn([1, cond_dim])
        self.null_cond_emb = paddle.create_parameter(
            shape=x.shape,
            dtype=str(x.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(
                x)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers
        self.downs = nn.LayerList([])
        self.ups = nn.LayerList([])
        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock3D, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.LayerList([
                    block_klass_cond(dim_in, dim_out),
                    block_klass_cond(dim_out, dim_out),
                    Residual(
                        PreNorm(
                            dim_out,
                            SpatialLinearAttention(dim_out, heads=attn_heads)))
                    if use_sparse_linear_attn else nn.Identity(),
                    Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                    Downsample(dim_out) if not is_last else nn.Identity()
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        spatial_attn = EinopsToAndFrom(
            # 'b c f h w', 'b f (h w) c',
            ['b', 'c', 'f', 'h', 'w'],
            ['b', 'f', 'hw', 'c'],
            SpatialTemporalAttention(mid_dim, heads=attn_heads))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.LayerList([
                    block_klass_cond(dim_out * 2, dim_in),
                    block_klass_cond(dim_in, dim_in),
                    Residual(
                        PreNorm(
                            dim_in,
                            SpatialLinearAttention(dim_in, heads=attn_heads)))
                    if use_sparse_linear_attn else nn.Identity(),
                    Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                    Upsample(dim_in) if not is_last else nn.Identity()
                ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(block_klass(dim * 2, dim),
                                        nn.Conv3D(dim, out_dim, 1))

    def forward_with_cond_scale(self, *args, cond_scale=2., **kwargs):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        prob_focus_present=0.,  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        return_dict=True,
    ):
        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        batch = x.shape[0]

        focus_present_mask = default(
            focus_present_mask, lambda: prob_mask_like(
                (batch, ), prob_focus_present))
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2])

        x = self.init_conv(x)

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance
        if self.has_cond:
            batch = x.shape[0]
            mask = prob_mask_like((batch, ), null_cond_prob)
            cond = paddle.where(paddle.reshape(mask, [mask.shape[0], 1]),
                                self.null_cond_emb, cond)
            t = paddle.concat((t, cond), axis=-1)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x,
                              pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x,
                                   pos_bias=time_rel_pos_bias,
                                   focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = paddle.concat((x, h.pop()), axis=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x,
                              pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = paddle.concat((x, r), axis=1)
        sample = self.final_conv(x)

        if not return_dict:
            return (sample, )
        return UNet3DOutput(sample=sample)
