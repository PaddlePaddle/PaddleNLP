# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

import unittest

import numpy as np
import paddle
from paddle import nn

from ppdiffusers.models.attention import (
    GEGLU,
    AdaLayerNorm,
    ApproximateGELU,
    AttentionBlock,
    Transformer2DModel,
)
from ppdiffusers.models.embeddings import get_timestep_embedding
from ppdiffusers.models.resnet import Downsample2D, Upsample2D


class EmbeddingsTests(unittest.TestCase):
    def test_timestep_embeddings(self):
        embedding_dim = 256
        timesteps = paddle.arange(16)

        t1 = get_timestep_embedding(timesteps, embedding_dim)

        # first vector should always be composed only of 0's and 1's
        assert (t1[0, : embedding_dim // 2] - 0).abs().sum() < 1e-5
        assert (t1[0, embedding_dim // 2 :] - 1).abs().sum() < 1e-5

        # last element of each vector should be one
        assert (t1[:, -1] - 1).abs().sum() < 1e-5

        # For large embeddings (e.g. 128) the frequency of every vector is higher
        # than the previous one which means that the gradients of later vectors are
        # ALWAYS higher than the previous ones
        grad_mean = np.abs(np.gradient(t1, axis=-1)).mean(axis=1)

        prev_grad = 0.0
        for grad in grad_mean:
            assert grad > prev_grad
            prev_grad = grad

    def test_timestep_defaults(self):
        embedding_dim = 16
        timesteps = paddle.arange(10)

        t1 = get_timestep_embedding(timesteps, embedding_dim)
        t2 = get_timestep_embedding(
            timesteps, embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=1, max_period=10_000
        )

        assert paddle.allclose(t1.cpu(), t2.cpu(), 1e-3)

    def test_timestep_flip_sin_cos(self):
        embedding_dim = 16
        timesteps = paddle.arange(10)

        t1 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True)
        t1 = paddle.concat([t1[:, embedding_dim // 2 :], t1[:, : embedding_dim // 2]], axis=-1)

        t2 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False)

        assert paddle.allclose(t1.cpu(), t2.cpu(), 1e-3)

    def test_timestep_downscale_freq_shift(self):
        embedding_dim = 16
        timesteps = paddle.arange(10)

        t1 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=0)
        t2 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=1)

        # get cosine half (vectors that are wrapped into cosine)
        cosine_half = (t1 - t2)[:, embedding_dim // 2 :]

        # cosine needs to be negative
        assert (np.abs((cosine_half <= 0).numpy()) - 1).sum() < 1e-5

    def test_sinoid_embeddings_hardcoded(self):
        embedding_dim = 64
        timesteps = paddle.arange(128)

        # standard unet, score_vde
        t1 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=1, flip_sin_to_cos=False)
        # glide, ldm
        t2 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=0, flip_sin_to_cos=True)
        # grad-tts
        t3 = get_timestep_embedding(timesteps, embedding_dim, scale=1000)

        assert paddle.allclose(
            t1[23:26, 47:50].flatten().cpu(),
            paddle.to_tensor([0.9646, 0.9804, 0.9892, 0.9615, 0.9787, 0.9882, 0.9582, 0.9769, 0.9872]),
            1e-3,
        )
        assert paddle.allclose(
            t2[23:26, 47:50].flatten().cpu(),
            paddle.to_tensor([0.3019, 0.2280, 0.1716, 0.3146, 0.2377, 0.1790, 0.3272, 0.2474, 0.1864]),
            1e-3,
        )
        assert paddle.allclose(
            t3[23:26, 47:50].flatten().cpu(),
            paddle.to_tensor([-0.9801, -0.9464, -0.9349, -0.3952, 0.8887, -0.9709, 0.5299, -0.2853, -0.9927]),
            1e-3,
        )


class Upsample2DBlockTests(unittest.TestCase):
    def test_upsample_default(self):
        paddle.seed(0)
        sample = paddle.randn([1, 32, 32, 32])
        upsample = Upsample2D(channels=32, use_conv=False)
        with paddle.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == [1, 32, 64, 64]
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [
                -1.5021564960479736,
                -0.12905766069889069,
                -0.12905766069889069,
                -1.9701517820358276,
                0.7877668738365173,
                0.7877668738365173,
                -1.9701517820358276,
                0.7877668738365173,
                0.7877668738365173,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_upsample_with_conv(self):
        paddle.seed(0)
        sample = paddle.randn([1, 32, 32, 32])
        upsample = Upsample2D(channels=32, use_conv=True)
        with paddle.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == [1, 32, 64, 64]
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [
                0.4583871364593506,
                -0.8221798539161682,
                -0.8228907585144043,
                0.3325321078300476,
                -0.24422502517700195,
                1.344732642173767,
                0.5239212512969971,
                -0.4814918637275696,
                0.17928099632263184,
            ]
        )

        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_upsample_with_conv_out_dim(self):
        paddle.seed(0)
        sample = paddle.randn([1, 32, 32, 32])
        upsample = Upsample2D(channels=32, use_conv=True, out_channels=64)
        with paddle.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == [1, 64, 64, 64]
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [
                0.9049283266067505,
                -1.6125869750976562,
                -1.0837469100952148,
                0.24520659446716309,
                -0.6669139266014099,
                0.5660533905029297,
                1.1056761741638184,
                2.1717309951782227,
                0.7197026610374451,
            ]
        )

        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_upsample_with_transpose(self):
        paddle.seed(0)
        sample = paddle.randn([1, 32, 32, 32])
        upsample = Upsample2D(channels=32, use_conv=False, use_conv_transpose=True)
        with paddle.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == [1, 32, 64, 64]
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [
                -0.05951346457004547,
                0.2695199251174927,
                0.26003628969192505,
                1.12237548828125,
                -0.07744795083999634,
                0.006375759840011597,
                0.6678807735443115,
                0.4432428181171417,
                -0.10978642851114273,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)


class Downsample2DBlockTests(unittest.TestCase):
    def test_downsample_default(self):
        paddle.seed(0)
        sample = paddle.randn([1, 32, 64, 64])
        downsample = Downsample2D(channels=32, use_conv=False)
        with paddle.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == [1, 32, 32, 32]
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [
                -0.24012964963912964,
                -0.034197285771369934,
                -1.0328047275543213,
                0.7861506938934326,
                -0.2086063176393509,
                -0.3999312222003937,
                0.25081655383110046,
                -0.23891538381576538,
                -1.4398303031921387,
            ]
        )
        max_diff = (output_slice.flatten() - expected_slice).abs().sum().item()
        assert max_diff <= 1e-3
        # assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-1)

    def test_downsample_with_conv(self):
        paddle.seed(0)
        sample = paddle.randn([1, 32, 64, 64])
        downsample = Downsample2D(channels=32, use_conv=True)
        with paddle.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == [1, 32, 32, 32]
        output_slice = downsampled[0, -1, -3:, -3:]

        expected_slice = paddle.to_tensor(
            [
                -0.009430217556655407,
                0.8657761216163635,
                1.7985490560531616,
                -0.61894291639328,
                -2.5752196311950684,
                1.2352519035339355,
                0.6046919822692871,
                -1.6499173641204834,
                -1.5272349119186401,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_downsample_with_conv_pad1(self):
        paddle.seed(0)
        sample = paddle.randn([1, 32, 64, 64])
        downsample = Downsample2D(channels=32, use_conv=True, padding=1)
        with paddle.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == [1, 32, 32, 32]
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [
                -0.009430217556655407,
                0.8657761216163635,
                1.7985490560531616,
                -0.61894291639328,
                -2.5752196311950684,
                1.2352519035339355,
                0.6046919822692871,
                -1.6499173641204834,
                -1.5272349119186401,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_downsample_with_conv_out_dim(self):
        paddle.seed(0)
        sample = paddle.randn([1, 32, 64, 64])
        downsample = Downsample2D(channels=32, use_conv=True, out_channels=16)
        with paddle.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == [1, 16, 32, 32]
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [
                0.10819266736507416,
                0.43043053150177,
                -0.7322822213172913,
                -1.923148512840271,
                1.0195047855377197,
                0.48796477913856506,
                1.6765365600585938,
                -4.072991847991943,
                0.8763526082038879,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)


class AttentionBlockTests(unittest.TestCase):
    def test_attention_block_default(self):
        paddle.seed(0)

        sample = paddle.randn([1, 32, 64, 64])
        attentionBlock = AttentionBlock(
            channels=32,
            num_head_channels=1,
            rescale_output_factor=1.0,
            eps=1e-6,
            norm_num_groups=32,
        )
        with paddle.no_grad():
            attention_scores = attentionBlock(sample)

        assert attention_scores.shape == [1, 32, 64, 64]
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = paddle.to_tensor(
            [
                1.63893962,
                -0.15776771,
                -1.11300254,
                -0.85402739,
                -0.56967819,
                -2.04937434,
                -0.37326077,
                -1.74031317,
                -0.52711684,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_attention_block_sd(self):
        # This version uses SD params and is compatible with mps
        paddle.seed(0)

        sample = paddle.randn([1, 512, 64, 64])
        attentionBlock = AttentionBlock(
            channels=512,
            rescale_output_factor=1.0,
            eps=1e-6,
            norm_num_groups=32,
        )
        with paddle.no_grad():
            attention_scores = attentionBlock(sample)

        assert attention_scores.shape == [1, 512, 64, 64]
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = paddle.to_tensor(
            [
                -0.80075705,
                -0.77035093,
                -3.52781916,
                -2.05402684,
                -0.77117395,
                -0.82782876,
                -0.48292717,
                1.60399365,
                0.62672436,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)


class Transformer2DModelTests(unittest.TestCase):
    def test_spatial_transformer_default(self):
        paddle.seed(0)

        sample = paddle.randn([1, 32, 64, 64])
        spatial_transformer_block = Transformer2DModel(
            in_channels=32,
            num_attention_heads=1,
            attention_head_dim=32,
            dropout=0.0,
            cross_attention_dim=None,
        )
        with paddle.no_grad():
            attention_scores = spatial_transformer_block(sample).sample

        assert attention_scores.shape == [1, 32, 64, 64]
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = paddle.to_tensor(
            [
                3.7865211963653564,
                -3.368237018585205,
                0.9498730897903442,
                3.3758397102355957,
                -2.4002809524536133,
                -4.587373733520508,
                -2.4826488494873047,
                -2.9742937088012695,
                -2.2530620098114014,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_spatial_transformer_cross_attention_dim(self):
        paddle.seed(0)

        sample = paddle.randn([1, 64, 64, 64])
        spatial_transformer_block = Transformer2DModel(
            in_channels=64,
            num_attention_heads=2,
            attention_head_dim=32,
            dropout=0.0,
            cross_attention_dim=64,
        )
        with paddle.no_grad():
            context = paddle.randn([1, 4, 64])
            attention_scores = spatial_transformer_block(sample, context).sample

        assert attention_scores.shape == [1, 64, 64, 64]
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = paddle.to_tensor(
            [
                -1.747774362564087,
                -5.564971446990967,
                -2.9772586822509766,
                0.22785288095474243,
                -0.05944812297821045,
                -1.3060683012008667,
                -0.8295403718948364,
                0.37562549114227295,
                -0.4789081811904907,
            ]
        )

        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_spatial_transformer_timestep(self):
        paddle.seed(0)

        num_embeds_ada_norm = 5

        sample = paddle.randn([1, 64, 64, 64])
        spatial_transformer_block = Transformer2DModel(
            in_channels=64,
            num_attention_heads=2,
            attention_head_dim=32,
            dropout=0.0,
            cross_attention_dim=64,
            num_embeds_ada_norm=num_embeds_ada_norm,
        )
        with paddle.no_grad():
            timestep_1 = paddle.to_tensor(1, dtype="int64")
            timestep_2 = paddle.to_tensor(2, dtype="int64")
            attention_scores_1 = spatial_transformer_block(sample, timestep=timestep_1).sample
            attention_scores_2 = spatial_transformer_block(sample, timestep=timestep_2).sample

        assert attention_scores_1.shape == [1, 64, 64, 64]
        assert attention_scores_2.shape == [1, 64, 64, 64]

        output_slice_1 = attention_scores_1[0, -1, -3:, -3:]
        output_slice_2 = attention_scores_2[0, -1, -3:, -3:]

        expected_slice_1 = paddle.to_tensor(
            [
                -0.8467671871185303,
                -1.8978886604309082,
                -4.740020751953125,
                -0.09714558720588684,
                -0.923940122127533,
                -0.09468311071395874,
                2.0457370281219482,
                -4.556329727172852,
                -0.6690530776977539,
            ]
        )
        expected_slice_2 = paddle.to_tensor(
            [
                -0.5957154035568237,
                -1.9309439659118652,
                -4.694167613983154,
                -0.18004107475280762,
                -0.8857038021087646,
                0.013831645250320435,
                2.0215866565704346,
                -4.488527774810791,
                -0.6237987279891968,
            ]
        )

        assert paddle.allclose(output_slice_1.flatten(), expected_slice_1, atol=1e-3)
        assert paddle.allclose(output_slice_2.flatten(), expected_slice_2, atol=1e-3)

    def test_spatial_transformer_dropout(self):
        paddle.seed(0)

        sample = paddle.randn([1, 32, 64, 64])
        spatial_transformer_block = Transformer2DModel(
            in_channels=32, num_attention_heads=2, attention_head_dim=16, dropout=0.3, cross_attention_dim=None
        )
        spatial_transformer_block.eval()

        with paddle.no_grad():
            attention_scores = spatial_transformer_block(sample).sample

        assert attention_scores.shape == [1, 32, 64, 64]
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = paddle.to_tensor(
            [
                3.6616106033325195,
                -3.0174713134765625,
                1.2159616947174072,
                3.594019889831543,
                -2.4868431091308594,
                -4.75314474105835,
                -2.537452220916748,
                -2.9906234741210938,
                -2.216505289077759,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_spatial_transformer_discrete(self):
        paddle.seed(0)

        num_embed = 5

        sample = paddle.randint(0, num_embed, (1, 32))
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1,
            attention_head_dim=32,
            num_vector_embeds=num_embed,
            sample_size=16,
        )

        spatial_transformer_block.eval()

        with paddle.no_grad():
            attention_scores = spatial_transformer_block(sample).sample

        assert attention_scores.shape == [1, num_embed - 1, 32]

        output_slice = attention_scores[0, -2:, -3:]

        expected_slice = paddle.to_tensor(
            [
                -1.8575713634490967,
                -1.9135738611221313,
                -1.8472729921340942,
                -0.5044772028923035,
                -2.0277059078216553,
                -3.0968878269195557,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_spatial_transformer_default_norm_layers(self):
        spatial_transformer_block = Transformer2DModel(num_attention_heads=1, attention_head_dim=32, in_channels=32)

        assert spatial_transformer_block.transformer_blocks[0].norm1.__class__ == nn.LayerNorm
        assert spatial_transformer_block.transformer_blocks[0].norm2.__class__ == nn.LayerNorm
        assert spatial_transformer_block.transformer_blocks[0].norm3.__class__ == nn.LayerNorm

    def test_spatial_transformer_ada_norm_layers(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1,
            attention_head_dim=32,
            in_channels=32,
            num_embeds_ada_norm=5,
        )

        assert spatial_transformer_block.transformer_blocks[0].norm1.__class__ == AdaLayerNorm
        assert spatial_transformer_block.transformer_blocks[0].norm2.__class__ == AdaLayerNorm
        assert spatial_transformer_block.transformer_blocks[0].norm3.__class__ == nn.LayerNorm

    def test_spatial_transformer_default_ff_layers(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1,
            attention_head_dim=32,
            in_channels=32,
        )

        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].__class__ == GEGLU
        assert spatial_transformer_block.transformer_blocks[0].ff.net[1].__class__ == nn.Dropout
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].__class__ == nn.Linear

        dim = 32
        inner_dim = 128

        # First dimension change
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.weight.shape[0] == dim
        # NOTE: inner_dim * 2 because GEGLU
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.weight.shape[1] == inner_dim * 2

        # Second dimension change
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].weight.shape[0] == inner_dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].weight.shape[1] == dim

    def test_spatial_transformer_geglu_approx_ff_layers(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1,
            attention_head_dim=32,
            in_channels=32,
            activation_fn="geglu-approximate",
        )

        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].__class__ == ApproximateGELU
        assert spatial_transformer_block.transformer_blocks[0].ff.net[1].__class__ == nn.Dropout
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].__class__ == nn.Linear

        dim = 32
        inner_dim = 128

        # First dimension change
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.weight.shape[0] == dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.weight.shape[1] == inner_dim

        # Second dimension change
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].weight.shape[0] == inner_dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].weight.shape[1] == dim

    def test_spatial_transformer_attention_bias(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1, attention_head_dim=32, in_channels=32, attention_bias=True
        )
        assert spatial_transformer_block.transformer_blocks[0].attn1.to_q._bias_attr is True
        assert spatial_transformer_block.transformer_blocks[0].attn1.to_k._bias_attr is True
        assert spatial_transformer_block.transformer_blocks[0].attn1.to_v._bias_attr is True
