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
from ppdiffusers_test.pipeline_params import (
    UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS,
    UNCONDITIONAL_IMAGE_GENERATION_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from ppdiffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from ppdiffusers.utils.testing_utils import require_paddle_gpu, slow


class DDIMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DDIMPipeline
    test_cpu_offload = False
    params = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "num_images_per_prompt",
        "latents",
        "callback",
        "callback_steps",
    }
    batch_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        scheduler = DDIMScheduler()
        components = {"unet": unet, "scheduler": scheduler}
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        inputs = {"batch_size": 1, "generator": generator, "num_inference_steps": 2, "output_type": "numpy"}
        return inputs

    def test_inference(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        self.assertEqual(image.shape, (1, 32, 32, 3))
        expected_slice = np.array([0.0, 0.00152004, 0.0, 0.0, 0.00860906, 0.00182715, 0.00189051, 1.0, 0.668702])
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 0.001)


@slow
@require_paddle_gpu
class DDIMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"
        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler()
        ddim = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddim.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = ddim(generator=generator, eta=0.0, output_type="numpy").images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.2060, 0.2042, 0.2022, 0.2193, 0.2146, 0.2110, 0.2471, 0.2446, 0.2388])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_inference_ema_bedroom(self):
        model_id = "google/ddpm-ema-bedroom-256"
        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler.from_pretrained(model_id)
        ddim = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddim.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = ddim(generator=generator, output_type="numpy").images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.1546, 0.1561, 0.1595, 0.1564, 0.1569, 0.1585, 0.1554, 0.1550, 0.1575])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
