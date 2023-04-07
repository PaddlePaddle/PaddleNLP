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
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionSAGPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusionSAGPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionSAGPipeline
    test_cpu_offload = False
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        paddle.seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        paddle.seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config).eval()
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": ".",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "sag_scale": 1.0,
            "output_type": "numpy",
        }
        return inputs


@slow
@require_paddle_gpu
class StableDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_1(self):
        sag_pipe = StableDiffusionSAGPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sag_pipe.set_progress_bar_config(disable=None)
        prompt = "."
        generator = paddle.Generator().manual_seed(0)
        output = sag_pipe(
            [prompt], generator=generator, guidance_scale=7.5, sag_scale=1.0, num_inference_steps=20, output_type="np"
        )
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.7477613, 0.76045597, 0.7464366, 0.778965, 0.75718963, 0.7487634, 0.77530396, 0.77426934, 0.7749926]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05

    def test_stable_diffusion_2(self):
        sag_pipe = StableDiffusionSAGPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sag_pipe.set_progress_bar_config(disable=None)
        prompt = "."
        generator = paddle.Generator().manual_seed(0)
        output = sag_pipe(
            [prompt], generator=generator, guidance_scale=7.5, sag_scale=1.0, num_inference_steps=20, output_type="np"
        )
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.8771595, 0.8521123, 0.8644101, 0.8680052, 0.8700466, 0.8897612, 0.87766427, 0.8636212, 0.86829203]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05
