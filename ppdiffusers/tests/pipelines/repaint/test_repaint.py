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
import unittest

import numpy as np
import paddle
from test_pipelines_common import PipelineTesterMixin

from ppdiffusers import RePaintPipeline, RePaintScheduler, UNet2DModel
from ppdiffusers.utils.testing_utils import load_image, load_numpy, nightly


class RepaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = RePaintPipeline

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        scheduler = RePaintScheduler()
        components = {"unet": unet, "scheduler": scheduler}
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        image = np.random.RandomState(seed).standard_normal((1, 3, 32, 32))
        image = paddle.to_tensor(image, dtype="float32")
        mask = (image > 0).cast(paddle.float32)
        inputs = {
            "image": image,
            "mask_image": mask,
            "generator": generator,
            "num_inference_steps": 5,
            "eta": 0.0,
            "jump_length": 2,
            "jump_n_sample": 2,
            "output_type": "numpy",
        }
        return inputs

    def test_repaint(self):
        components = self.get_dummy_components()
        sd_pipe = RePaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [
                0.08341649174690247,
                0.5426262617111206,
                0.5497109889984131,
                0.009035259485244751,
                0.0,
                1.0,
                0.05136755108833313,
                0.56046462059021,
                0.6273577809333801,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


@nightly
class RepaintPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_celebahq(self):
        original_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/celeba_hq_256.png")
        mask_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/mask_256.png")
        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests" "/repaint/celeba_hq_256_result.npy"
        )

        model_id = "google/ddpm-ema-celebahq-256"
        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = RePaintScheduler.from_pretrained(model_id)

        repaint = RePaintPipeline(unet=unet, scheduler=scheduler)
        repaint.set_progress_bar_config(disable=None)
        repaint.enable_attention_slicing()

        generator = paddle.Generator().manual_seed(0)
        output = repaint(
            original_image,
            mask_image,
            num_inference_steps=250,
            eta=0.0,
            jump_length=10,
            jump_n_sample=10,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (256, 256, 3)
        assert np.abs(expected_image - image).mean() < 1e-2
