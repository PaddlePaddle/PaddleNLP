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

import random
import unittest

import numpy as np
import paddle
from test_pipelines_common import PipelineTesterMixin

from ppdiffusers import DDIMScheduler, LDMSuperResolutionPipeline, UNet2DModel, VQModel
from ppdiffusers.utils import PIL_INTERPOLATION, floats_tensor, load_image, slow


class LDMSuperResolutionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = (32, 32)

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
        image = ldm(init_image, generator=generator, num_inference_steps=2, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.12982192635536194,
                0.8338450193405151,
                0.46506837010383606,
                0.5459575653076172,
                0.666222095489502,
                0.38444048166275024,
                0.7219546437263489,
                0.571929931640625,
                0.36579442024230957,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


@slow
class LDMSuperResolutionPipelineIntegrationTests(unittest.TestCase):
    def test_inference_superresolution(self):
        init_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/teddy_bear_pool.png")
        init_image = init_image.resize((64, 64), resample=PIL_INTERPOLATION["lanczos"])

        ldm = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
        ldm.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        image = ldm(init_image, generator=generator, num_inference_steps=20, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.7418, 0.7472, 0.7424, 0.7422, 0.7463, 0.726, 0.7382, 0.7248, 0.6828])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
