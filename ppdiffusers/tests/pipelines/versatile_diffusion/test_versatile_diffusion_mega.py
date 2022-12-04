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

from ppdiffusers import VersatileDiffusionPipeline
from ppdiffusers.utils.testing_utils import load_image, slow


class VersatileDiffusionMegaPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pass


@slow
class VersatileDiffusionMegaPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_from_save_pretrained(self):
        pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.set_progress_bar_config(disable=None)

        prompt_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg")

        generator = paddle.Generator().manual_seed(0)
        image = pipe.dual_guided(
            prompt="first prompt",
            image=prompt_image,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        ).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = VersatileDiffusionPipeline.from_pretrained(tmpdirname)
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        new_image = pipe.dual_guided(
            prompt="first prompt",
            image=prompt_image,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        ).images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't have the same forward pass"

    def test_inference_dual_guided_then_text_to_image(self):
        pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.set_progress_bar_config(disable=None)

        prompt = "cyberpunk 2077"
        init_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg")
        generator = paddle.Generator().manual_seed(0)
        image = pipe.dual_guided(
            prompt=prompt,
            image=init_image,
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
                0.06040886044502258,
                0.0689929723739624,
                0.074072927236557,
                0.06452780961990356,
                0.07012578845024109,
                0.0790989100933075,
                0.07237845659255981,
                0.07687109708786011,
                0.08553361892700195,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

        prompt = "A painting of a squirrel eating a burger "
        generator = paddle.Generator().manual_seed(0)
        image = pipe.text_to_image(
            prompt=prompt, generator=generator, guidance_scale=7.5, num_inference_steps=50, output_type="numpy"
        ).images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [
                0.040520429611206055,
                0.01816403865814209,
                0.0,
                0.03902044892311096,
                0.004770994186401367,
                0.045984357595443726,
                0.04142877459526062,
                0.0,
                0.02198156714439392,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
