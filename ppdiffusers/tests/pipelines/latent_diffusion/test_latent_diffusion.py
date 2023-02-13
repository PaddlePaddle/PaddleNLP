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

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LDMTextToImagePipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils.testing_utils import load_numpy, nightly, slow


class LDMTextToImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = LDMTextToImagePipeline

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
        text_encoder_config = dict(
            text_embed_dim=32,
            text_heads=4,
            text_layers=5,
            vocab_size=1000,
        )
        text_encoder_config = CLIPTextConfig.from_dict(text_encoder_config)
        text_encoder = CLIPTextModel(text_encoder_config)
        text_encoder.eval()
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vqvae": vae,
            "bert": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_inference_text2img(self):

        components = self.get_dummy_components()
        pipe = LDMTextToImagePipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.23783081769943237,
                0.2683892548084259,
                0.3141400218009949,
                0.2046811282634735,
                0.35299813747406006,
                0.5098875761032104,
                0.1809311807155609,
                0.22014355659484863,
                0.37353450059890747,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


@slow
class LDMTextToImagePipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 32, 32))
        latents = paddle.to_tensor(latents, dtype=dtype)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_ldm_default_ddim(self):
        pipe = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array(
            [
                0.3991875648498535,
                0.410067081451416,
                0.3996868431568146,
                0.4110398292541504,
                0.3848368227481842,
                0.40485692024230957,
                0.40426763892173767,
                0.39793622493743896,
                0.42825278639793396,
            ]
        )
        max_diff = np.abs(expected_slice - image_slice).max()
        assert max_diff < 1e-3


@nightly
class LDMTextToImagePipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 32, 32))
        latents = paddle.to_tensor(latents, dtype=dtype)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_ldm_default_ddim(self):
        pipe = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests/ldm_text2img/ldm_large_256_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
