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
from test_pipelines_common import PipelineTesterMixin

from ppdiffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from ppdiffusers.utils import deprecate
from ppdiffusers.utils.testing_utils import slow


class DDPMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    @property
    def dummy_uncond_unet(self):
        paddle.seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

    def test_inference(self):
        unet = self.dummy_uncond_unet
        scheduler = DDPMScheduler()

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="numpy").images

        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = ddpm(generator=generator, num_inference_steps=2, output_type="numpy", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [
                8.296966552734375e-05,
                0.635993480682373,
                0.5485098361968994,
                0.0648922324180603,
                0.44553104043006897,
                0.21740025281906128,
                0.2344886064529419,
                0.9999170303344727,
                0.7372267842292786,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_deprecated_predict_epsilon(self):
        deprecate("remove this test", "0.10.0", "remove")
        unet = self.dummy_uncond_unet
        scheduler = DDPMScheduler(predict_epsilon=False)

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="numpy").images

        generator = paddle.Generator().manual_seed(0)
        image_eps = ddpm(generator=generator, num_inference_steps=2, output_type="numpy", predict_epsilon=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_eps_slice = image_eps[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        tolerance = 3e-2
        assert np.abs(image_slice.flatten() - image_eps_slice.flatten()).max() < tolerance

    def test_inference_predict_sample(self):
        unet = self.dummy_uncond_unet
        scheduler = DDPMScheduler(prediction_type="sample")

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="numpy").images

        generator = paddle.Generator().manual_seed(0)
        image_eps = ddpm(generator=generator, num_inference_steps=2, output_type="numpy")[0]

        image_slice = image[0, -3:, -3:, -1]
        image_eps_slice = image_eps[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        tolerance = 1e-2
        assert np.abs(image_slice.flatten() - image_eps_slice.flatten()).max() < tolerance


@slow
class DDPMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDPMScheduler.from_pretrained(model_id)

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4454, 0.2025, 0.0315, 0.3023, 0.2575, 0.1031, 0.0953, 0.1604, 0.2020])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
