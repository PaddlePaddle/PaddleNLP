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

from ppdiffusers import VersatileDiffusionPipeline
from ppdiffusers.utils.testing_utils import load_image, slow

from test_pipelines_common import PipelineTesterMixin


class VersatileDiffusionMegaPipelineFastTests(PipelineTesterMixin,
                                              unittest.TestCase):
    pass


@slow
class VersatileDiffusionMegaPipelineIntegrationTests(unittest.TestCase):

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_from_pretrained_save_pretrained(self):
        pipe = VersatileDiffusionPipeline.from_pretrained(
            "shi-labs/versatile-diffusion")
        pipe.set_progress_bar_config(disable=None)

        prompt_image = load_image(
            "https://raw.githubusercontent.com/SHI-Labs/Versatile-Diffusion/master/assets/benz.jpg"
        )

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

        generator = generator.manual_seed(0)
        new_image = pipe.dual_guided(
            prompt="first prompt",
            image=prompt_image,
            text_to_image_strength=0.75,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="numpy",
        ).images

        assert np.abs(image - new_image).sum(
        ) < 1e-5, "Models don't have the same forward pass"

    def test_inference_dual_guided_then_text_to_image(self):
        pipe = VersatileDiffusionPipeline.from_pretrained(
            "shi-labs/versatile-diffusion")
        pipe.set_progress_bar_config(disable=None)

        prompt = "cyberpunk 2077"
        init_image = load_image(
            "https://raw.githubusercontent.com/SHI-Labs/Versatile-Diffusion/master/assets/benz.jpg"
        )
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
        expected_slice = np.array([
            0.014, 0.0112, 0.0136, 0.0145, 0.0107, 0.0113, 0.0272, 0.0215,
            0.0216
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

        prompt = "A painting of a squirrel eating a burger "
        generator = paddle.Generator().manual_seed(0)
        image = pipe.text_to_image(prompt=prompt,
                                   generator=generator,
                                   guidance_scale=7.5,
                                   num_inference_steps=50,
                                   output_type="numpy").images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.0408, 0.0181, 0.0, 0.0388, 0.0046, 0.0461, 0.0411, 0.0, 0.0222])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

        pipe = VersatileDiffusionPipeline.from_pretrained(
            "shi-labs/versatile-diffusion", torch_dtype=torch.float16)
        image = pipe.image_variation(init_image,
                                     generator=generator,
                                     output_type="numpy").images[0]

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([
            0.0657, 0.0529, 0.0455, 0.0802, 0.0570, 0.0179, 0.0267, 0.0483,
            0.0769
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
