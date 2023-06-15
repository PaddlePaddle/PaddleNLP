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
    LDMTextToImagePipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils.testing_utils import (
    load_numpy,
    nightly,
    require_paddle_gpu,
    slow,
)


class LDMTextToImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = LDMTextToImagePipeline
    params = TEXT_TO_IMAGE_PARAMS - {
        "negative_prompt",
        "negative_prompt_embeds",
        "cross_attention_kwargs",
        "prompt_embeds",
    }
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "num_images_per_prompt",
        "callback",
        "callback_steps",
    }
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    test_cpu_offload = False

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
            block_out_channels=(32, 64),
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
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
        components = {"unet": unet, "scheduler": scheduler, "vqvae": vae, "bert": text_encoder, "tokenizer": tokenizer}
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
        # chan
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.29159048, 0.20539099, 0.29126638, 0.19384867, 0.2436865, 0.45562512, 0.12645364, 0.14380667, 0.3520335]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001


@slow
@require_paddle_gpu
class LDMTextToImagePipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed=seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 32, 32))
        latents = paddle.to_tensor(latents).cast(dtype)
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
        expected_slice = np.array([0.51825, 0.5285, 0.52543, 0.54258, 0.52304, 0.52569, 0.54363, 0.55276, 0.56878])
        max_diff = np.abs(expected_slice - image_slice).max()
        assert max_diff < 0.02


@nightly
@require_paddle_gpu
class LDMTextToImagePipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed=seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 32, 32))
        latents = paddle.to_tensor(latents).cast(dtype)
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
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/ldm_text2img/ldm_large_256_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.05
