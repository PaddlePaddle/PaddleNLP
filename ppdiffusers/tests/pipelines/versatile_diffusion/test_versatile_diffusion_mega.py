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

from ppdiffusers import VersatileDiffusionPipeline
from ppdiffusers.utils.testing_utils import load_image, require_paddle_gpu, slow


class VersatileDiffusionMegaPipelineFastTests(unittest.TestCase):
    pass


@slow
@require_paddle_gpu
class VersatileDiffusionMegaPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_from_save_pretrained(self):
        pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.set_progress_bar_config(disable=None)
        prompt_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/versatile_diffusion/benz.jpg"
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
            pipe = VersatileDiffusionPipeline.from_pretrained(tmpdirname, from_diffusers=False)
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
        assert np.abs(image - new_image).sum() < 1e-05, "Models don't have the same forward pass"

    def test_inference_dual_guided_then_text_to_image(self):
        pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion", paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        prompt = "cyberpunk 2077"
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/versatile_diffusion/benz.jpg"
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
        # breakpoint()
        # expected_slice = np.array([0.1448, 0.1619, 0.1741, 0.1086, 0.1147, 0.1128, 0.1199, 0.1165, 0.1001])
        expected_slice = np.array(
            [
                0.03100586,
                0.02929688,
                0.03271484,
                0.02807617,
                0.02905273,
                0.03173828,
                0.02685547,
                0.02807617,
                0.03271484,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
        prompt = "A painting of a squirrel eating a burger "
        generator = paddle.Generator().manual_seed(0)
        image = pipe.text_to_image(
            prompt=prompt, generator=generator, guidance_scale=7.5, num_inference_steps=50, output_type="numpy"
        ).images
        image_slice = image[0, 253:256, 253:256, -1]
        assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.3367, 0.3169, 0.2656, 0.387, 0.479, 0.3796, 0.4009, 0.4878, 0.4778])
        expected_slice = np.array(
            [0.0390625, 0.00854492, 0.0, 0.03930664, 0.00878906, 0.04711914, 0.03686523, 0.0, 0.0246582]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
        image = pipe.image_variation(init_image, generator=generator, output_type="numpy").images
        image_slice = image[0, 253:256, 253:256, -1]
        assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.3076, 0.3123, 0.3284, 0.3782, 0.377, 0.3894, 0.4297, 0.4331, 0.4456])
        expected_slice = np.array(
            [0.34472656, 0.1940918, 0.10546875, 0.38134766, 0.24560547, 0.13208008, 0.38867188, 0.30566406, 0.18188477]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
