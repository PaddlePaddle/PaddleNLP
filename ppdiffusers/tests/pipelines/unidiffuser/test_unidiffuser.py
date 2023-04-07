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

import gc
import tempfile
import unittest

import numpy as np
import paddle

from ppdiffusers import UniDiffuserPipeline
from ppdiffusers.utils.testing_utils import load_image, require_paddle_gpu, slow


class UniDiffuserPipelineFastTests(unittest.TestCase):
    pass


@slow
@require_paddle_gpu
class UniDiffuserPipelinePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_from_save_pretrained(self):
        pipe = UniDiffuserPipeline.from_pretrained("thu-ml/unidiffuser")

        pipe.set_progress_bar_config(disable=None)
        prompt_image = load_image("https://bj.bcebos.com/v1/paddlenlp/models/community/thu-ml/data/space.jpg")
        generator = paddle.Generator().manual_seed(0)
        image = pipe(mode="i2t2i", image=prompt_image, generator=generator, output_type="numpy").images
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = UniDiffuserPipeline.from_pretrained(tmpdirname, from_diffusers=False)

        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        new_image = pipe(mode="i2t2i", image=prompt_image, generator=generator, output_type="numpy").images

        assert np.abs(image - new_image).sum() < 1e-05, "Models don't have the same forward pass"

    def test_inference(self):
        pipe = UniDiffuserPipeline.from_pretrained("thu-ml/unidiffuser", paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        prompt = "An astronaut floating out of space into the Earth"
        init_image = load_image("https://bj.bcebos.com/v1/paddlenlp/models/community/thu-ml/data/space.jpg")
        generator = paddle.Generator().manual_seed(0)
        image = pipe(mode="t2i", prompt=prompt, generator=generator, output_type="numpy").images
        image_slice = image[0, 253:256, 253:256, -1]
        assert image.shape == (1, 512, 512, 3)
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
