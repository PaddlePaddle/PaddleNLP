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

from ppdiffusers import VersatileDiffusionImageVariationPipeline
from ppdiffusers.utils.testing_utils import load_image, slow


class VersatileDiffusionImageVariationPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pass


@slow
class VersatileDiffusionImageVariationPipelineIntegrationTests(unittest.TestCase):
    def test_inference_image_variations(self):
        pipe = VersatileDiffusionImageVariationPipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.set_progress_bar_config(disable=None)

        image_prompt = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg")
        generator = paddle.Generator().manual_seed(0)
        image = pipe(
            image=image_prompt,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=50,
            output_type="numpy",
        ).images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [
                0.011769682168960571,
                0.22415882349014282,
                0.40258049964904785,
                0.08735564351081848,
                0.08671644330024719,
                0.27234160900115967,
                0.26517266035079956,
                0.0,
                0.10928627848625183,
            ]
        )
        print(image_slice.flatten().tolist())
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
