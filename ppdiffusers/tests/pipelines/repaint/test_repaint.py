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

from ppdiffusers import RePaintPipeline, RePaintScheduler, UNet2DModel
from ppdiffusers.utils.testing_utils import load_image, slow


@slow
class RepaintPipelineIntegrationTests(unittest.TestCase):
    def test_celebahq(self):
        original_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/celeba_hq_256.png")
        mask_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/mask_256.png")
        expected_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/celeba_hq_256_result.png"
        )
        expected_image = np.array(expected_image, dtype=np.float32) / 255.0

        model_id = "google/ddpm-ema-celebahq-256"
        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = RePaintScheduler.from_pretrained(model_id)

        repaint = RePaintPipeline(unet=unet, scheduler=scheduler)

        generator = paddle.Generator().manual_seed(0)
        output = repaint(
            original_image,
            mask_image,
            num_inference_steps=250,
            eta=0.0,
            jump_length=10,
            jump_n_sample=10,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (256, 256, 3)
        assert np.abs(expected_image - image).mean() < 1e-2
