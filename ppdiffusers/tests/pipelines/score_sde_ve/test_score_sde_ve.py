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

from ppdiffusers import ScoreSdeVePipeline, ScoreSdeVeScheduler, UNet2DModel
from ppdiffusers.utils.testing_utils import require_paddle, slow


class ScoreSdeVeipelineFastTests(unittest.TestCase):
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
        scheduler = ScoreSdeVeScheduler()
        sde_ve = ScoreSdeVePipeline(unet=unet, scheduler=scheduler)
        sde_ve.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = sde_ve(num_inference_steps=2, output_type="numpy", generator=generator).images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sde_ve(num_inference_steps=2, output_type="numpy", generator=generator, return_dict=False)[
            0
        ]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01


@slow
@require_paddle
class ScoreSdeVePipelineIntegrationTests(unittest.TestCase):
    def test_inference(self):
        model_id = "google/ncsnpp-church-256"
        model = UNet2DModel.from_pretrained(model_id)
        scheduler = ScoreSdeVeScheduler.from_pretrained(model_id)
        sde_ve = ScoreSdeVePipeline(unet=model, scheduler=scheduler)
        sde_ve.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = sde_ve(num_inference_steps=10, output_type="numpy", generator=generator).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
