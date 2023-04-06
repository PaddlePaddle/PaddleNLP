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
import unittest

import numpy as np
import paddle
from ppdiffusers_test.pipeline_params import (
    IMAGE_INPAINTING_BATCH_PARAMS,
    IMAGE_INPAINTING_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from ppdiffusers import RePaintPipeline, RePaintScheduler, UNet2DModel
from ppdiffusers.utils.testing_utils import (
    load_image,
    load_numpy,
    nightly,
    require_paddle_gpu,
)


class RepaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = RePaintPipeline
    test_cpu_offload = False
    params = IMAGE_INPAINTING_PARAMS - {"width", "height", "guidance_scale"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "latents",
        "num_images_per_prompt",
        "callback",
        "callback_steps",
    }
    batch_params = IMAGE_INPAINTING_BATCH_PARAMS

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
        image = paddle.to_tensor(data=image).cast("float32")
        mask = (image > 0).cast("float32")
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
            [0.08341709, 0.54262626, 0.549711, 0.00903523, 0.0, 1.0, 0.05136755, 0.5604646, 0.6273578]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    # RePaint can hardly be made deterministic since the scheduler is currently always
    # nondeterministic
    @unittest.skip("non-deterministic pipeline")
    def test_inference_batch_single_identical(self):
        return super().test_inference_batch_single_identical()


@nightly
@require_paddle_gpu
class RepaintPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_celebahq(self):
        original_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask_256.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256_result.npy"
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
        assert np.abs(expected_image - image).mean() < 0.01
