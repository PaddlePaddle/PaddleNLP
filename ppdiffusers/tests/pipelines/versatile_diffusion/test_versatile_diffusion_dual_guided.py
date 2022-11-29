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

import gc
import tempfile
import unittest

import numpy as np
import paddle
from test_pipelines_common import PipelineTesterMixin

from ppdiffusers import VersatileDiffusionDualGuidedPipeline
from ppdiffusers.utils.testing_utils import load_image, slow


class VersatileDiffusionDualGuidedPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pass


@slow
class VersatileDiffusionDualGuidedPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_remove_unused_weights_save_load(self):
        pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion")
        # remove text_unet
        pipe.remove_unused_weights()
        pipe.set_progress_bar_config(disable=None)

        second_prompt = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg")

        generator = paddle.Generator().manual_seed(0)
        image = pipe(
            prompt="first prompt",
            image=second_prompt,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        ).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(tmpdirname)

        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        new_image = pipe(
            prompt="first prompt",
            image=second_prompt,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        ).images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't have the same forward pass"

    def test_inference_dual_guided(self):
        pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.remove_unused_weights()
        pipe.set_progress_bar_config(disable=None)

        first_prompt = "cyberpunk 2077"
        second_prompt = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg")
        generator = paddle.Generator().manual_seed(0)
        image = pipe(
            prompt=first_prompt,
            image=second_prompt,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=50,
            output_type="numpy",
        ).images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [
                0.05819129943847656,
                0.06465867161750793,
                0.0698845386505127,
                0.06031566858291626,
                0.0696127712726593,
                0.07399758696556091,
                0.06906205415725708,
                0.07634878158569336,
                0.0813780426979065,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
