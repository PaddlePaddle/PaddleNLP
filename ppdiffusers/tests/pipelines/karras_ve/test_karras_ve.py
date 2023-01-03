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

from ppdiffusers import KarrasVePipeline, KarrasVeScheduler, UNet2DModel
from ppdiffusers.utils.testing_utils import slow


class KarrasVePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
        expected_slice = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
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
            [
                0.7528197765350342,
                0.7529420852661133,
                0.7601380944252014,
                0.7548195123672485,
                0.7569247484207153,
                0.7577681541442871,
                0.7605229616165161,
                0.7589468955993652,
                0.759920597076416,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
