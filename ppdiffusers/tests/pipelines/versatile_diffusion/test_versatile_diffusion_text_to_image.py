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

from ppdiffusers import VersatileDiffusionTextToImagePipeline
from ppdiffusers.utils.testing_utils import slow


class VersatileDiffusionTextToImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pass


@slow
class VersatileDiffusionTextToImagePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_remove_unused_weights_save_load(self):
        pipe = VersatileDiffusionTextToImagePipeline.from_pretrained("shi-labs/versatile-diffusion")
        # remove text_unet
        pipe.remove_unused_weights()
        pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger "
        generator = paddle.Generator().manual_seed(0)
        image = pipe(
            prompt=prompt, generator=generator, guidance_scale=7.5, num_inference_steps=2, output_type="numpy"
        ).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = VersatileDiffusionTextToImagePipeline.from_pretrained(tmpdirname)
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        new_image = pipe(
            prompt=prompt, generator=generator, guidance_scale=7.5, num_inference_steps=2, output_type="numpy"
        ).images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't have the same forward pass"

    def test_inference_text2img(self):
        pipe = VersatileDiffusionTextToImagePipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger "
        generator = paddle.Generator().manual_seed(0)
        image = pipe(
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
