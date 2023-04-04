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
import random
import unittest

import numpy as np
import paddle
from ppdiffusers_test.pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    CycleDiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class CycleDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = CycleDiffusionPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {
        "negative_prompt",
        "height",
        "width",
        "negative_prompt_embeds",
    }
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

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
            num_train_timesteps=1000,
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
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": "An astronaut riding an elephant",
            "source_prompt": "An astronaut riding a horse",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "eta": 0.1,
            "strength": 0.8,
            "guidance_scale": 3,
            "source_guidance_scale": 1,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_cycle(self):
        components = self.get_dummy_components()
        pipe = CycleDiffusionPipeline(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)
        images = output.images
        image_slice = images[0, -3:, -3:, -1]
        assert images.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [0.04812625, 0.77983606, 0.71009433, 0.15924984, 0.9788434, 0.49732354, 0.362224, 0.6481595, 0.4530744]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_cycle_fp16(self):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "to"):
                components[name] = module.to(dtype=paddle.float16)
        pipe = CycleDiffusionPipeline(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)
        images = output.images
        image_slice = images[0, -3:, -3:, -1]
        assert images.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [0.05053711, 0.78125, 0.7114258, 0.15991211, 0.9785156, 0.49804688, 0.36279297, 0.6484375, 0.45361328]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    @unittest.skip("non-deterministic pipeline")
    def test_inference_batch_single_identical(self):
        return super().test_inference_batch_single_identical()


@slow
@require_paddle_gpu
class CycleDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_cycle_diffusion_pipeline_fp16(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/cycle-diffusion/black_colored_car.png"
        )
        expected_image = np.array([[0.14477539, 0.20483398, 0.14135742], [0.10009766, 0.17602539, 0.11083984]])
        init_image = init_image.resize((512, 512))
        model_id = "CompVis/stable-diffusion-v1-4"
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = CycleDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, safety_checker=None, paddle_dtype=paddle.float16, revision="fp16"
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        source_prompt = "A black colored car"
        prompt = "A blue colored car"
        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt,
            source_prompt=source_prompt,
            image=init_image,
            num_inference_steps=100,
            eta=0.1,
            strength=0.85,
            guidance_scale=3,
            source_guidance_scale=1,
            generator=generator,
            output_type="np",
        )
        image = output.images
        assert np.abs(image[0][0][:2] - expected_image).max() < 0.5

    def test_cycle_diffusion_pipeline(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/cycle-diffusion/black_colored_car.png"
        )
        expected_image = np.array([[0.16294342, 0.20514232, 0.14554858], [0.11476257, 0.16831946, 0.11495486]])
        init_image = init_image.resize((512, 512))
        model_id = "CompVis/stable-diffusion-v1-4"
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = CycleDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        source_prompt = "A black colored car"
        prompt = "A blue colored car"
        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt,
            source_prompt=source_prompt,
            image=init_image,
            num_inference_steps=100,
            eta=0.1,
            strength=0.85,
            guidance_scale=3,
            source_guidance_scale=1,
            generator=generator,
            output_type="np",
        )
        image = output.images
        assert np.abs(image[0][0][:2] - expected_image).max() < 0.01
