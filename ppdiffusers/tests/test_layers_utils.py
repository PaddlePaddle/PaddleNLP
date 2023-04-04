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

import unittest

import numpy as np
import paddle
import paddle.nn

from ppdiffusers.models.attention import (
    GEGLU,
    AdaLayerNorm,
    ApproximateGELU,
    AttentionBlock,
)
from ppdiffusers.models.embeddings import get_timestep_embedding
from ppdiffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from ppdiffusers.models.transformer_2d import Transformer2DModel


class EmbeddingsTests(unittest.TestCase):
    def test_timestep_embeddings(self):
        embedding_dim = 256
        timesteps = paddle.arange(start=16)
        t1 = get_timestep_embedding(timesteps, embedding_dim)
        assert (t1[0, : embedding_dim // 2] - 0).abs().sum() < 1e-05
        assert (t1[0, embedding_dim // 2 :] - 1).abs().sum() < 1e-05
        assert (t1[:, -1] - 1).abs().sum() < 1e-05
        grad_mean = np.abs(np.gradient(t1, axis=-1)).mean(axis=1)
        prev_grad = 0.0
        for grad in grad_mean:
            assert grad > prev_grad
            prev_grad = grad

    def test_timestep_defaults(self):
        embedding_dim = 16
        timesteps = paddle.arange(start=10)
        t1 = get_timestep_embedding(timesteps, embedding_dim)
        t2 = get_timestep_embedding(
            timesteps, embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=1, max_period=10000
        )
        assert paddle.allclose(t1.cpu(), t2.cpu(), atol=0.01)

    def test_timestep_flip_sin_cos(self):
        embedding_dim = 16
        timesteps = paddle.arange(start=10)
        t1 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True)
        t1 = paddle.concat(x=[t1[:, embedding_dim // 2 :], t1[:, : embedding_dim // 2]], axis=-1)
        t2 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False)
        assert paddle.allclose(t1.cpu(), t2.cpu(), atol=0.01)

    def test_timestep_downscale_freq_shift(self):
        embedding_dim = 16
        timesteps = paddle.arange(start=10)
        t1 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=0)
        t2 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=1)
        cosine_half = (t1 - t2)[:, embedding_dim // 2 :]
        assert (np.abs((cosine_half <= 0).numpy()) - 1).sum() < 1e-05

    def test_sinoid_embeddings_hardcoded(self):
        embedding_dim = 64
        timesteps = paddle.arange(start=128)
        t1 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=1, flip_sin_to_cos=False)
        t2 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=0, flip_sin_to_cos=True)
        t3 = get_timestep_embedding(timesteps, embedding_dim, scale=1000)
        assert paddle.allclose(
            t1[23:26, 47:50].flatten().cpu(),
            paddle.to_tensor([0.9646, 0.9804, 0.9892, 0.9615, 0.9787, 0.9882, 0.9582, 0.9769, 0.9872]),
            atol=0.01,
        )
        assert paddle.allclose(
            t2[23:26, 47:50].flatten().cpu(),
            paddle.to_tensor([0.3019, 0.228, 0.1716, 0.3146, 0.2377, 0.179, 0.3272, 0.2474, 0.1864]),
            atol=0.01,
        )
        assert paddle.allclose(
            t3[23:26, 47:50].flatten().cpu(),
            paddle.to_tensor([-0.9801, -0.9464, -0.9349, -0.3952, 0.8887, -0.9709, 0.5299, -0.2853, -0.9927]),
            atol=0.01,
        )


class Upsample2DBlockTests(unittest.TestCase):
    def test_upsample_default(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 32, 32])
        upsample = Upsample2D(channels=32, use_conv=False)
        with paddle.no_grad():
            upsampled = upsample(sample)
        assert tuple(upsampled.shape) == (1, 32, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [
                -1.50215650,
                -0.12905766,
                -0.12905766,
                -1.97015178,
                0.78776687,
                0.78776687,
                -1.97015178,
                0.78776687,
                0.78776687,
            ]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_upsample_with_conv(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 32, 32])
        upsample = Upsample2D(channels=32, use_conv=True)
        with paddle.no_grad():
            upsampled = upsample(sample)
        assert tuple(upsampled.shape) == (1, 32, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor([0.7145, 1.3773, 0.3492, 0.8448, 1.0839, -0.3341, 0.5956, 0.125, -0.4841])
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_upsample_with_conv_out_dim(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 32, 32])
        upsample = Upsample2D(channels=32, use_conv=True, out_channels=64)
        with paddle.no_grad():
            upsampled = upsample(sample)
        assert tuple(upsampled.shape) == (1, 64, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor([0.2703, 0.1656, -0.2538, -0.0553, -0.2984, 0.1044, 0.1155, 0.2579, 0.7755])
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_upsample_with_transpose(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 32, 32])
        upsample = Upsample2D(channels=32, use_conv=False, use_conv_transpose=True)
        with paddle.no_grad():
            upsampled = upsample(sample)
        assert tuple(upsampled.shape) == (1, 32, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [-0.3028, -0.1582, 0.0071, 0.035, -0.4799, -0.1139, 0.1056, -0.1153, -0.1046]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)


class Downsample2DBlockTests(unittest.TestCase):
    def test_downsample_default(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        downsample = Downsample2D(channels=32, use_conv=False)
        with paddle.no_grad():
            downsampled = downsample(sample)
        assert tuple(downsampled.shape) == (1, 32, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor([-0.0513, -0.3889, 0.064, 0.0836, -0.546, -0.0341, -0.0169, -0.6967, 0.1179])
        max_diff = (output_slice.flatten() - expected_slice).abs().sum().item()
        assert max_diff <= 0.001

    def test_downsample_with_conv(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        downsample = Downsample2D(channels=32, use_conv=True)
        with paddle.no_grad():
            downsampled = downsample(sample)
        assert tuple(downsampled.shape) == (1, 32, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [0.9267, 0.5878, 0.3337, 1.2321, -0.1191, -0.3984, -0.7532, -0.0715, -0.3913]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_downsample_with_conv_pad1(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        downsample = Downsample2D(channels=32, use_conv=True, padding=1)
        with paddle.no_grad():
            downsampled = downsample(sample)
        assert tuple(downsampled.shape) == (1, 32, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [0.9267, 0.5878, 0.3337, 1.2321, -0.1191, -0.3984, -0.7532, -0.0715, -0.3913]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_downsample_with_conv_out_dim(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        downsample = Downsample2D(channels=32, use_conv=True, out_channels=16)
        with paddle.no_grad():
            downsampled = downsample(sample)
        assert tuple(downsampled.shape) == (1, 16, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor([-0.6586, 0.5985, 0.0721, 0.1256, -0.1492, 0.4436, -0.2544, 0.5021, 1.1522])
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)


class ResnetBlock2DTests(unittest.TestCase):
    def test_resnet_default(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        temb = paddle.randn(shape=[1, 128])
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128)
        with paddle.no_grad():
            output_tensor = resnet_block(sample, temb)
        assert tuple(output_tensor.shape) == (1, 32, 64, 64)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [-1.901, -0.2974, -0.8245, -1.3533, 0.8742, -0.9645, -2.0584, 1.3387, -0.4746]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_restnet_with_use_in_shortcut(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        temb = paddle.randn(shape=[1, 128])
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, use_in_shortcut=True)
        with paddle.no_grad():
            output_tensor = resnet_block(sample, temb)
        assert tuple(output_tensor.shape) == (1, 32, 64, 64)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor([0.2226, -1.0791, -0.1629, 0.3659, -0.2889, -1.2376, 0.0582, 0.9206, 0.0044])
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_resnet_up(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        temb = paddle.randn(shape=[1, 128])
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, up=True)
        with paddle.no_grad():
            output_tensor = resnet_block(sample, temb)
        assert tuple(output_tensor.shape) == (1, 32, 128, 128)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [1.213, -0.8753, -0.9027, 1.5783, -0.5362, -0.5001, 1.0726, -0.7732, -0.4182]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_resnet_down(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        temb = paddle.randn(shape=[1, 128])
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, down=True)
        with paddle.no_grad():
            output_tensor = resnet_block(sample, temb)
        assert tuple(output_tensor.shape) == (1, 32, 32, 32)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [-0.3002, -0.7135, 0.1359, 0.0561, -0.7935, 0.0113, -0.1766, -0.6714, -0.0436]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_restnet_with_kernel_fir(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        temb = paddle.randn(shape=[1, 128])
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, kernel="fir", down=True)
        with paddle.no_grad():
            output_tensor = resnet_block(sample, temb)
        assert tuple(output_tensor.shape) == (1, 32, 32, 32)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [-0.0934, -0.5729, 0.0909, -0.271, -0.5044, 0.0243, -0.0665, -0.5267, -0.3136]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_restnet_with_kernel_sde_vp(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        temb = paddle.randn(shape=[1, 128])
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, kernel="sde_vp", down=True)
        with paddle.no_grad():
            output_tensor = resnet_block(sample, temb)
        assert tuple(output_tensor.shape) == (1, 32, 32, 32)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [-0.3002, -0.7135, 0.1359, 0.0561, -0.7935, 0.0113, -0.1766, -0.6714, -0.0436]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)


class AttentionBlockTests(unittest.TestCase):
    def test_attention_block_default(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        attentionBlock = AttentionBlock(
            channels=32, num_head_channels=1, rescale_output_factor=1.0, eps=1e-06, norm_num_groups=32
        )
        with paddle.no_grad():
            attention_scores = attentionBlock(sample)
        assert attention_scores.shape == [1, 32, 64, 64]
        output_slice = attention_scores[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [-1.4975, -0.0038, -0.7847, -1.4567, 1.122, -0.8962, -1.7394, 1.1319, -0.5427]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_attention_block_sd(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 512, 64, 64])
        attentionBlock = AttentionBlock(channels=512, rescale_output_factor=1.0, eps=1e-06, norm_num_groups=32)
        with paddle.no_grad():
            attention_scores = attentionBlock(sample)
        assert attention_scores.shape == [1, 512, 64, 64]
        output_slice = attention_scores[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [-0.6621, -0.0156, -3.2766, 0.8025, -0.8609, 0.282, 0.0905, -1.1179, -3.2126]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)


class Transformer2DModelTests(unittest.TestCase):
    def test_spatial_transformer_default(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        spatial_transformer_block = Transformer2DModel(
            in_channels=32, num_attention_heads=1, attention_head_dim=32, dropout=0.0, cross_attention_dim=None
        )
        with paddle.no_grad():
            attention_scores = spatial_transformer_block(sample).sample
        assert attention_scores.shape == [1, 32, 64, 64]
        output_slice = attention_scores[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [-1.9455, -0.0066, -1.3933, -1.5878, 0.5325, -0.6486, -1.8648, 0.7515, -0.9689]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_spatial_transformer_cross_attention_dim(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 64, 64, 64])
        spatial_transformer_block = Transformer2DModel(
            in_channels=64, num_attention_heads=2, attention_head_dim=32, dropout=0.0, cross_attention_dim=64
        )
        with paddle.no_grad():
            context = paddle.randn(shape=[1, 4, 64])
            attention_scores = spatial_transformer_block(sample, context).sample
        assert attention_scores.shape == [1, 64, 64, 64]
        output_slice = attention_scores[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [-0.2555, -0.8877, -2.4739, -2.2251, 1.2714, 0.0807, -0.4161, -1.6408, -0.0471]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_spatial_transformer_timestep(self):
        paddle.seed(0)
        num_embeds_ada_norm = 5
        sample = paddle.randn(shape=[1, 64, 64, 64])
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
        assert tuple(attention_scores_1.shape) == (1, 64, 64, 64)
        assert tuple(attention_scores_2.shape) == (1, 64, 64, 64)
        output_slice_1 = attention_scores_1[0, -1, -3:, -3:]
        output_slice_2 = attention_scores_2[0, -1, -3:, -3:]
        expected_slice_1 = paddle.to_tensor(
            [-0.1874, -0.9704, -1.429, -1.3357, 1.5138, 0.3036, -0.0976, -1.1667, 0.1283]
        )
        expected_slice_2 = paddle.to_tensor(
            [-0.3493, -1.0924, -1.6161, -1.5016, 1.4245, 0.1367, -0.2526, -1.3109, -0.0547]
        )
        assert paddle.allclose(output_slice_1.flatten(), expected_slice_1, atol=0.01)
        assert paddle.allclose(output_slice_2.flatten(), expected_slice_2, atol=0.01)

    def test_spatial_transformer_dropout(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        spatial_transformer_block = Transformer2DModel(
            in_channels=32, num_attention_heads=2, attention_head_dim=16, dropout=0.3, cross_attention_dim=None
        ).eval()
        with paddle.no_grad():
            attention_scores = spatial_transformer_block(sample).sample
        assert attention_scores.shape == [1, 32, 64, 64]
        output_slice = attention_scores[0, -1, -3:, -3:]
        expected_slice = paddle.to_tensor(
            [-1.938, -0.0083, -1.3771, -1.5819, 0.5209, -0.6441, -1.8545, 0.7563, -0.9615]
        )
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_spatial_transformer_discrete(self):
        paddle.seed(0)
        num_embed = 5
        sample = paddle.randint(0, num_embed, (1, 32))
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1, attention_head_dim=32, num_vector_embeds=num_embed, sample_size=16
        ).eval()
        with paddle.no_grad():
            attention_scores = spatial_transformer_block(sample).sample
        assert attention_scores.shape == [1, num_embed - 1, 32]
        output_slice = attention_scores[0, -2:, -3:]
        expected_slice = paddle.to_tensor([-1.7648, -1.0241, -2.0985, -1.8035, -1.6404, -1.2098])
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_spatial_transformer_default_norm_layers(self):
        spatial_transformer_block = Transformer2DModel(num_attention_heads=1, attention_head_dim=32, in_channels=32)
        assert spatial_transformer_block.transformer_blocks[0].norm1.__class__ == paddle.nn.LayerNorm
        assert spatial_transformer_block.transformer_blocks[0].norm3.__class__ == paddle.nn.LayerNorm

    def test_spatial_transformer_ada_norm_layers(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1, attention_head_dim=32, in_channels=32, num_embeds_ada_norm=5
        )
        assert spatial_transformer_block.transformer_blocks[0].norm1.__class__ == AdaLayerNorm
        assert spatial_transformer_block.transformer_blocks[0].norm3.__class__ == paddle.nn.LayerNorm

    def test_spatial_transformer_default_ff_layers(self):
        spatial_transformer_block = Transformer2DModel(num_attention_heads=1, attention_head_dim=32, in_channels=32)
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].__class__ == GEGLU
        assert spatial_transformer_block.transformer_blocks[0].ff.net[1].__class__ == paddle.nn.Dropout
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].__class__ == paddle.nn.Linear
        dim = 32
        inner_dim = 128
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.weight.shape[0] == dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.weight.shape[1] == inner_dim * 2
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].weight.shape[0] == inner_dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].weight.shape[1] == dim

    def test_spatial_transformer_geglu_approx_ff_layers(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1, attention_head_dim=32, in_channels=32, activation_fn="geglu-approximate"
        )
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].__class__ == ApproximateGELU
        assert spatial_transformer_block.transformer_blocks[0].ff.net[1].__class__ == paddle.nn.Dropout
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].__class__ == paddle.nn.Linear
        dim = 32
        inner_dim = 128
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.weight.shape[0] == dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.weight.shape[1] == inner_dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].weight.shape[0] == inner_dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].weight.shape[1] == dim

    def test_spatial_transformer_attention_bias(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1, attention_head_dim=32, in_channels=32, attention_bias=True
        )
        assert spatial_transformer_block.transformer_blocks[0].attn1.to_q.bias is not None
        assert spatial_transformer_block.transformer_blocks[0].attn1.to_k.bias is not None
        assert spatial_transformer_block.transformer_blocks[0].attn1.to_v.bias is not None
