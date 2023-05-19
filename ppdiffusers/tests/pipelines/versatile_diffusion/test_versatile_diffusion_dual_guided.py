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

import gc
import tempfile
import unittest

import numpy as np
import paddle

from ppdiffusers import VersatileDiffusionDualGuidedPipeline
from ppdiffusers.utils.testing_utils import load_image, require_paddle_gpu, slow


class VersatileDiffusionDualGuidedPipelineFastTests(unittest.TestCase):
    pass


@slow
@require_paddle_gpu
class VersatileDiffusionDualGuidedPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_remove_unused_weights_save_load(self):
        pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.remove_unused_weights()
        pipe.set_progress_bar_config(disable=None)
        second_prompt = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/versatile_diffusion/benz.jpg"
        )
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
            pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(tmpdirname, from_diffusers=False)
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
        assert np.abs(image - new_image).sum() < 1e-05, "Models don't have the same forward pass"

    def test_inference_dual_guided(self):
        pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.remove_unused_weights()
        pipe.set_progress_bar_config(disable=None)
        first_prompt = "cyberpunk 2077"
        second_prompt = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/versatile_diffusion/benz.jpg"
        )
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
                0.01500076,
                0.01142624,
                0.01418972,
                0.01518875,
                0.01114869,
                0.01190853,
                0.02978998,
                0.02376354,
                0.02396089,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
