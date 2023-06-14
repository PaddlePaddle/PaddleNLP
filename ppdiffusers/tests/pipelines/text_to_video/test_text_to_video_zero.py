# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from ppdiffusers import DDIMScheduler, TextToVideoZeroPipeline
from ppdiffusers.utils import load_pd, require_paddle_gpu, slow

from ..test_pipelines_common import assert_mean_pixel_difference


@slow
@require_paddle_gpu
class TextToVideoZeroPipelineSlowTests(unittest.TestCase):
    def test_full_model(self):
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype="float16")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        generator = paddle.Generator().manual_seed(0)
        prompt = "A bear is playing a guitar on Times Square"
        result = pipe(prompt=prompt, generator=generator).images
        expected_result = load_pd(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text-to-video/A bear is playing a guitar on Times Square.pt"
        )
        assert_mean_pixel_difference(result, expected_result)
