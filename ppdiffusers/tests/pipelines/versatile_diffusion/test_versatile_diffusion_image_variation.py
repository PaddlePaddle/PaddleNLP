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

import unittest

import numpy as np
import paddle

from ppdiffusers import VersatileDiffusionImageVariationPipeline
from ppdiffusers.utils.testing_utils import load_image, require_paddle_gpu, slow


class VersatileDiffusionImageVariationPipelineFastTests(unittest.TestCase):
    pass


@slow
@require_paddle_gpu
class VersatileDiffusionImageVariationPipelineIntegrationTests(unittest.TestCase):
    def test_inference_image_variations(self):
        pipe = VersatileDiffusionImageVariationPipeline.from_pretrained("shi-labs/versatile-diffusion")
        pipe.set_progress_bar_config(disable=None)
        image_prompt = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/versatile_diffusion/benz.jpg"
        )
        generator = paddle.Generator().manual_seed(0)
        image = pipe(
            image=image_prompt, generator=generator, guidance_scale=7.5, num_inference_steps=50, output_type="numpy"
        ).images
        image_slice = image[0, 253:256, 253:256, -1]
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.12047189, 0.19138041, 0.22884357, 0.08833978, 0.1594424, 0.16826832, 0.07032129, 0.14926612, 0.12981007]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
