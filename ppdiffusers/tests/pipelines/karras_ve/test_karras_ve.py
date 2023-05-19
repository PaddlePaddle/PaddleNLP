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

from ppdiffusers import KarrasVePipeline, KarrasVeScheduler, UNet2DModel
from ppdiffusers.utils.testing_utils import require_paddle, slow


class KarrasVePipelineFastTests(unittest.TestCase):
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
        scheduler = KarrasVeScheduler()
        pipe = KarrasVePipeline(unet=unet, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = pipe(num_inference_steps=2, generator=generator, output_type="numpy").images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = pipe(num_inference_steps=2, generator=generator, output_type="numpy", return_dict=False)[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01


@slow
@require_paddle
class KarrasVePipelineIntegrationTests(unittest.TestCase):
    def test_inference(self):
        model_id = "google/ncsnpp-celebahq-256"
        model = UNet2DModel.from_pretrained(model_id)
        scheduler = KarrasVeScheduler()
        pipe = KarrasVePipeline(unet=model, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = pipe(num_inference_steps=20, generator=generator, output_type="numpy").images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array(
            [0.7528239, 0.7529462, 0.76014197, 0.75482357, 0.75692874, 0.7577723, 0.760527, 0.758951, 0.7599246]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
