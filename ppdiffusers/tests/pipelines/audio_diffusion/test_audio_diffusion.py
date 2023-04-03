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

import gc
import unittest

import numpy as np
import paddle

from ppdiffusers import (
    AudioDiffusionPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    Mel,
    UNet2DConditionModel,
    UNet2DModel,
)
from ppdiffusers.utils import slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class PipelineFastTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @property
    def dummy_unet(self):
        paddle.seed(0)
        model = UNet2DModel(
            sample_size=(32, 64),
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128),
            down_block_types=("AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D"),
        )
        return model

    @property
    def dummy_unet_condition(self):
        paddle.seed(0)
        model = UNet2DConditionModel(
            sample_size=(64, 32),
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128),
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=10,
        )
        return model

    @property
    def dummy_vqvae_and_unet(self):
        paddle.seed(0)
        vqvae = AutoencoderKL(
            sample_size=(128, 64),
            in_channels=1,
            out_channels=1,
            latent_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128),
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        )
        unet = UNet2DModel(
            sample_size=(64, 32),
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128),
            down_block_types=("AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D"),
        )
        return vqvae, unet

    def test_audio_diffusion(self):
        mel = Mel()
        scheduler = DDPMScheduler()
        pipe = AudioDiffusionPipeline(vqvae=None, unet=self.dummy_unet, mel=mel, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(42)
        output = pipe(generator=generator, steps=4)
        audio = output.audios[0]
        image = output.images[0]
        generator = paddle.Generator().manual_seed(42)
        output = pipe(generator=generator, steps=4, return_dict=False)
        image_from_tuple = output[0][0]
        assert audio.shape == (1, (self.dummy_unet.sample_size[1] - 1) * mel.hop_length)
        assert image.height == self.dummy_unet.sample_size[0] and image.width == self.dummy_unet.sample_size[1]
        image_slice = np.frombuffer(image.tobytes(), dtype="uint8")[:10]
        image_from_tuple_slice = np.frombuffer(image_from_tuple.tobytes(), dtype="uint8")[:10]
        expected_slice = np.array([0, 252, 0, 160, 144, 1, 0, 211, 99, 3])
        assert np.abs(image_slice.flatten() - expected_slice).max() == 0
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() == 0
        scheduler = DDIMScheduler()
        dummy_vqvae_and_unet = self.dummy_vqvae_and_unet
        pipe = AudioDiffusionPipeline(
            vqvae=self.dummy_vqvae_and_unet[0], unet=dummy_vqvae_and_unet[1], mel=mel, scheduler=scheduler
        )
        pipe.set_progress_bar_config(disable=None)
        np.random.seed(0)
        raw_audio = np.random.uniform(-1, 1, ((dummy_vqvae_and_unet[0].sample_size[1] - 1) * mel.hop_length,))
        generator = paddle.Generator().manual_seed(42)
        output = pipe(raw_audio=raw_audio, generator=generator, start_step=5, steps=10)
        image = output.images[0]
        assert (
            image.height == self.dummy_vqvae_and_unet[0].sample_size[0]
            and image.width == self.dummy_vqvae_and_unet[0].sample_size[1]
        )
        image_slice = np.frombuffer(image.tobytes(), dtype="uint8")[:10]
        expected_slice = np.array([128, 100, 153, 95, 92, 77, 130, 121, 81, 166])
        assert np.abs(image_slice.flatten() - expected_slice).max() == 0
        dummy_unet_condition = self.dummy_unet_condition
        pipe = AudioDiffusionPipeline(
            vqvae=self.dummy_vqvae_and_unet[0], unet=dummy_unet_condition, mel=mel, scheduler=scheduler
        )
        np.random.seed(0)
        encoding = paddle.rand(shape=(1, 1, 10))
        output = pipe(generator=generator, encoding=encoding)
        image = output.images[0]
        image_slice = np.frombuffer(image.tobytes(), dtype="uint8")[:10]
        expected_slice = np.array([166, 156, 163, 149, 148, 149, 145, 139, 157, 141])
        assert np.abs(image_slice.flatten() - expected_slice).max() == 0


@slow
@require_paddle_gpu
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_audio_diffusion(self):
        pipe = DiffusionPipeline.from_pretrained("teticio/audio-diffusion-ddim-256")
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(42)
        output = pipe(generator=generator)
        audio = output.audios[0]
        image = output.images[0]
        assert audio.shape == (1, (pipe.unet.sample_size[1] - 1) * pipe.mel.hop_length)
        assert image.height == pipe.unet.sample_size[0] and image.width == pipe.unet.sample_size[1]
        image_slice = np.frombuffer(image.tobytes(), dtype="uint8")[:10]
        expected_slice = np.array([151, 167, 154, 144, 122, 134, 121, 105, 70, 26])
        assert np.abs(image_slice.flatten() - expected_slice).max() == 0
