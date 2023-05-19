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

from ppdiffusers import VersatileDiffusionTextToImagePipeline
from ppdiffusers.utils.testing_utils import nightly, require_paddle_gpu


class VersatileDiffusionTextToImagePipelineFastTests(unittest.TestCase):
    pass


@nightly
@require_paddle_gpu
class VersatileDiffusionTextToImagePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_remove_unused_weights_save_load(self):
        pipe = VersatileDiffusionTextToImagePipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.remove_unused_weights()
        pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger "
        generator = paddle.Generator().manual_seed(0)
        image = pipe(
            prompt=prompt, generator=generator, guidance_scale=7.5, num_inference_steps=2, output_type="numpy"
        ).images
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = VersatileDiffusionTextToImagePipeline.from_pretrained(tmpdirname, from_diffusers=False)
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        new_image = pipe(
            prompt=prompt, generator=generator, guidance_scale=7.5, num_inference_steps=2, output_type="numpy"
        ).images
        assert np.abs(image - new_image).sum() < 1e-05, "Models don't have the same forward pass"

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
        # expected_slice = np.array([0.3493, 0.3757, 0.4093, 0.4495, 0.4233, 0.4102, 0.4507, 0.4756, 0.4787])
        expected_slice = np.array(
            [0.0390625, 0.00854492, 0.0, 0.03930664, 0.00878906, 0.04711914, 0.03686523, 0.0, 0.0246582]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
