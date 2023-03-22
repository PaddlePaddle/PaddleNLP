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

import random
import unittest

import numpy as np
import paddle

from ppdiffusers import DDIMScheduler, LDMSuperResolutionPipeline, UNet2DModel, VQModel
from ppdiffusers.utils import PIL_INTERPOLATION, floats_tensor, load_image, slow
from ppdiffusers.utils.testing_utils import require_paddle


class LDMSuperResolutionPipelineFastTests(unittest.TestCase):
    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = 32, 32
        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
        return image

    @property
    def dummy_uncond_unet(self):
        paddle.seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=6,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

    @property
    def dummy_vq_model(self):
        paddle.seed(0)
        model = VQModel(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=3,
        )
        return model

    def test_inference_superresolution(self):
        unet = self.dummy_uncond_unet
        scheduler = DDIMScheduler()
        vqvae = self.dummy_vq_model
        ldm = LDMSuperResolutionPipeline(unet=unet, vqvae=vqvae, scheduler=scheduler)
        ldm.set_progress_bar_config(disable=None)
        init_image = self.dummy_image
        generator = paddle.Generator().manual_seed(0)
        image = ldm(image=init_image, generator=generator, num_inference_steps=2, output_type="numpy").images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.12982202, 0.8338444, 0.46506804, 0.5459576, 0.6662215, 0.38444045, 0.72195464, 0.5719301, 0.36579454]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_inference_superresolution_fp16(self):
        unet = self.dummy_uncond_unet
        scheduler = DDIMScheduler()
        vqvae = self.dummy_vq_model
        unet = unet.to(dtype=paddle.float16)
        vqvae = vqvae.to(dtype=paddle.float16)
        ldm = LDMSuperResolutionPipeline(unet=unet, vqvae=vqvae, scheduler=scheduler)
        ldm.set_progress_bar_config(disable=None)
        init_image = self.dummy_image
        image = ldm(init_image, num_inference_steps=2, output_type="numpy").images
        assert image.shape == (1, 64, 64, 3)


@slow
@require_paddle
class LDMSuperResolutionPipelineIntegrationTests(unittest.TestCase):
    def test_inference_superresolution(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/vq_diffusion/teddy_bear_pool.png"
        )
        init_image = init_image.resize((64, 64), resample=PIL_INTERPOLATION["lanczos"])
        ldm = LDMSuperResolutionPipeline.from_pretrained("duongna/ldm-super-resolution")
        ldm.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = ldm(image=init_image, generator=generator, num_inference_steps=20, output_type="numpy").images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.7644, 0.7679, 0.7642, 0.7633, 0.7666, 0.756, 0.7425, 0.7257, 0.6907])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05
