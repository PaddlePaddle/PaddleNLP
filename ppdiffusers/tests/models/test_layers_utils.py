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
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_upsample_with_conv_out_dim(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 32, 32])
        upsample = Upsample2D(channels=32, use_conv=True, out_channels=64)
        with paddle.no_grad():
            upsampled = upsample(sample)
        assert tuple(upsampled.shape) == (1, 64, 64, 64)
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
            [
                -0.05951342731714249,
                0.26951998472213745,
                0.2600363492965698,
                1.12237548828125,
                -0.07744798064231873,
                0.006375734228640795,
                0.6678807735443115,
                0.44324278831481934,
                -0.10978640615940094,
            ]
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
        assert paddle.allclose(output_slice.flatten(), expected_slice, atol=0.01)

    def test_downsample_with_conv_out_dim(self):
        paddle.seed(0)
        sample = paddle.randn(shape=[1, 32, 64, 64])
        downsample = Downsample2D(channels=32, use_conv=True, out_channels=16)
        with paddle.no_grad():
            downsampled = downsample(sample)
        assert tuple(downsampled.shape) == (1, 16, 32, 32)
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
            [
                1.9816107749938965,
                1.4443503618240356,
                -1.0354782342910767,
                0.23985600471496582,
                -1.0868161916732788,
                -1.5830397605895996,
                -0.041037797927856445,
                -1.2574901580810547,
                -0.5504958629608154,
            ]
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
        expected_slice = paddle.to_tensor(
            [
                -0.9861348867416382,
                -1.097771406173706,
                0.268703430891037,
                0.40997087955474854,
                -4.26219367980957,
                1.758486270904541,
                -0.8979732990264893,
                0.30774950981140137,
                3.2780206203460693,
            ]
        )
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
            [
                0.2874237298965454,
                -2.6432056427001953,
                -2.1900298595428467,
                -0.48899877071380615,
                -1.1637755632400513,
                -1.084446907043457,
                -1.1333439350128174,
                0.2726985812187195,
                -0.014697253704071045,
            ]
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
            [
                1.54087495803833,
                0.26700693368911743,
                -0.540952742099762,
                2.7190208435058594,
                -0.09766747057437897,
                0.23407122492790222,
                0.47980907559394836,
                0.6348602771759033,
                -0.75424242019653322,
            ]
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
            [
                0.9914248585700989,
                0.4773162007331848,
                -0.021942138671875,
                2.482321262359619,
                0.18839354813098907,
                0.1516135334968567,
                0.7221578359603882,
                0.3920581340789795,
                -0.24661940336227417,
            ]
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
            [
                1.54087495803833,
                0.26700693368911743,
                -0.540952742099762,
                2.7190208435058594,
                -0.09766747057437897,
                0.23407122492790222,
                0.47980907559394836,
                0.6348602771759033,
                -0.7542424201965332,
            ]
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
            [
                1.638939619064331,
                -0.15776772797107697,
                -1.1130025386810303,
                -0.8540273904800415,
                -0.5696781873703003,
                -2.0493741035461426,
                -0.3732607960700989,
                -1.740313172340393,
                -0.5271167755126953,
            ]
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
            [
                -0.8007570505142212,
                -0.770350992679596,
                -3.5278191566467285,
                -2.0540268421173096,
                -0.7711739540100098,
                -0.8278288245201111,
                -0.48292720317840576,
                1.6039936542510986,
                0.626724362373352,
            ]
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
            [
                2.6310853958129883,
                5.990478515625,
                0.5715246200561523,
                -2.5269505977630615,
                -2.853764057159424,
                -5.163403511047363,
                0.2880846858024597,
                -5.925153732299805,
                2.316770076751709,
            ]
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
            [
                -0.08756911754608154,
                -3.94197940826416,
                -0.25678586959838867,
                2.1481714248657227,
                2.327033042907715,
                0.29948690533638,
                1.3845969438552856,
                0.7825677394866943,
                1.4856826066970825,
            ]
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
            [
                -0.15322405099868774,
                -1.265586018562317,
                -5.424124717712402,
                -0.7333418130874634,
                -0.5904415249824524,
                0.9293081760406494,
                1.1033945083618164,
                -5.200987815856934,
                -0.7598087787628174,
            ]
        )
        expected_slice_2 = paddle.to_tensor(
            [
                0.12572699785232544,
                -1.0498149394989014,
                -5.207070350646973,
                -0.41757693886756897,
                -0.25374162197113037,
                1.152648687362671,
                1.422953724861145,
                -4.933906078338623,
                -0.564710259437561,
            ]
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
            [
                2.535370349884033,
                6.2350993156433105,
                0.8244613409042358,
                -2.6684911251068115,
                -2.758057117462158,
                -5.176937103271484,
                0.3372979760169983,
                -5.837750434875488,
                2.3483340740203857,
            ]
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
        expected_slice = paddle.to_tensor(
            [
                -0.14130862057209015,
                -0.14278407394886017,
                -0.498604953289032,
                -3.2408740520477295,
                -3.852043390274048,
                -2.099970579147339,
            ]
        )
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
