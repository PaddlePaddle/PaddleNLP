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

from functools import partial
from dataclasses import dataclass

import paddle
from paddle import nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from ..utils import BaseOutput
from .embeddings import RelativePositionBias
from .attention import SpatialTemporalAttention, SpatialLinearAttention
from .rotary_embedding import RotaryEmbedding
from .resnet import Residual, PreNorm, ResnetBlock3D
from .video_utils import default, is_odd, prob_mask_like, Downsample, Upsample, EinopsToAndFrom


@dataclass
class DecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Decoded output sample of the model. Output of the last layer of the model.
    """

    sample: paddle.Tensor


class Encoder(nn.Layer):

    def __init__(self,
                 dim,
                 z_channels,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 attn_heads=8,
                 attn_dim_head=32,
                 init_dim=None,
                 init_kernel_size=3,
                 use_sparse_linear_attn=True,
                 block_type='resnet',
                 resnet_groups=8):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom(
            'b c f h w', 'b (h w) f c',
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

        # layers
        self.downs = nn.LayerList([])
        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock3D, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=None)

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
            'b c f h w', 'b f (h w) c',
            SpatialTemporalAttention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        self.final_conv = nn.Sequential(block_klass(mid_dim, mid_dim),
                                        nn.Conv3D(mid_dim, z_channels, 1))

    def forward(
        self,
        x,
        focus_present_mask=None,
        prob_focus_present=0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        batch = x.shape[0]

        focus_present_mask = default(
            focus_present_mask, lambda: prob_mask_like(
                (batch, ), prob_focus_present))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2])

        x = self.init_conv(x)

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        t = None

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

        return self.final_conv(x)


class Decoder(nn.Layer):

    def __init__(self,
                 dim,
                 z_channels,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 attn_heads=8,
                 attn_dim_head=32,
                 init_dim=None,
                 init_kernel_size=3,
                 use_sparse_linear_attn=True,
                 block_type='resnet',
                 resnet_groups=8):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom(
            'b c f h w', 'b (h w) f c',
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
        self.init_conv = nn.Conv3D(z_channels,
                                   init_dim * dim_mults[-1],
                                   (1, init_kernel_size, init_kernel_size),
                                   padding=(0, init_padding, init_padding))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.ups = nn.LayerList([])
        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock3D, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=None)

        # modules for all layers
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c',
            SpatialTemporalAttention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.LayerList([
                    block_klass_cond(dim_out, dim_in),
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
        self.final_conv = block_klass(dim, dim)
        self.conv_out = nn.Conv3D(dim, out_dim, 1)

    def forward(
        self,
        x,
        focus_present_mask=None,
        prob_focus_present=0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        batch = x.shape[0]

        focus_present_mask = default(
            focus_present_mask, lambda: prob_mask_like(
                (batch, ), prob_focus_present))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2])

        x = self.init_conv(x)
        t = None

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x,
                                   pos_bias=time_rel_pos_bias,
                                   focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x,
                              pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = self.final_conv(x)
        return self.conv_out(x)


class AutoencoderKLVid(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        dim,
        z_channels,
        dim_mults,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="video",
        monitor=None,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(dim=dim,
                               z_channels=z_channels,
                               dim_mults=dim_mults)
        self.decoder = Decoder(dim=dim,
                               z_channels=z_channels,
                               dim_mults=dim_mults)
        self.quant_conv = nn.Conv3D(z_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv3D(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        h = self.encoder(x)
        x = self.quant_conv(h)
        return x

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return DecoderOutput(sample=dec)

    def forward(self, input, sample_posterior=True):
        z = self.encode(input)
        dec = self.decode(z)
        return dec, None
