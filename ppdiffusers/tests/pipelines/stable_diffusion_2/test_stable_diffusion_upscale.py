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
from PIL import Image

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionUpscalePipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusionUpscalePipelineFastTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = 32, 32
        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
        return image

    @property
    def dummy_cond_unet_upscale(self):
        paddle.seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=7,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            attention_head_dim=8,
            use_linear_projection=True,
            only_cross_attention=(True, True, False),
            num_class_embeds=100,
        )
        return model

    @property
    def dummy_vae(self):
        paddle.seed(0)
        model = AutoencoderKL(
            block_out_channels=[32, 32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        return model

    @property
    def dummy_text_encoder(self):
        paddle.seed(0)
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=512,
        )
        return CLIPTextModel(config).eval()

    def test_stable_diffusion_upscale(self):
        unet = self.dummy_cond_unet_upscale
        low_res_scheduler = DDPMScheduler()
        scheduler = DDIMScheduler(prediction_type="v_prediction")
        vae = self.dummy_vae
        text_encoder = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        image = self.dummy_image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        low_res_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        sd_pipe = StableDiffusionUpscalePipeline(
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            max_noise_level=350,
        )
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe(
            [prompt],
            image=low_res_image,
            generator=generator,
            guidance_scale=6.0,
            noise_level=20,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            image=low_res_image,
            generator=generator,
            guidance_scale=6.0,
            noise_level=20,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        expected_height_width = low_res_image.size[0] * 4
        assert image.shape == (1, expected_height_width, expected_height_width, 3)
        expected_slice = np.array([0.2562, 0.3606, 0.4204, 0.4469, 0.4822, 0.4647, 0.5315, 0.5748, 0.5606])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_upscale_batch(self):
        unet = self.dummy_cond_unet_upscale
        low_res_scheduler = DDPMScheduler()
        scheduler = DDIMScheduler(prediction_type="v_prediction")
        vae = self.dummy_vae
        text_encoder = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        image = self.dummy_image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        low_res_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        sd_pipe = StableDiffusionUpscalePipeline(
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            max_noise_level=350,
        )
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        output = sd_pipe(
            2 * [prompt],
            image=2 * [low_res_image],
            guidance_scale=6.0,
            noise_level=20,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images
        assert image.shape[0] == 2
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe(
            [prompt],
            image=low_res_image,
            generator=generator,
            num_images_per_prompt=2,
            guidance_scale=6.0,
            noise_level=20,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images
        assert image.shape[0] == 2

    def test_stable_diffusion_upscale_fp16(self):
        """Test that stable diffusion upscale works with fp16"""
        unet = self.dummy_cond_unet_upscale
        low_res_scheduler = DDPMScheduler()
        scheduler = DDIMScheduler(prediction_type="v_prediction")
        vae = self.dummy_vae
        text_encoder = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        image = self.dummy_image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        low_res_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        unet = unet.to(dtype=paddle.float16)
        text_encoder = text_encoder.to(dtype=paddle.float16)
        sd_pipe = StableDiffusionUpscalePipeline(
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            max_noise_level=350,
        )
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        image = sd_pipe(
            [prompt], image=low_res_image, generator=generator, num_inference_steps=2, output_type="np"
        ).images
        expected_height_width = low_res_image.size[0] * 4
        assert image.shape == (1, expected_height_width, expected_height_width, 3)


@slow
@require_paddle_gpu
class StableDiffusionUpscalePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_upscale_pipeline(self):
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
        )
        # invalid expected_image
        # expected_image = load_numpy(
        #     'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/upsampled_cat.npy'
        #     )
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        prompt = "a cat sitting on a park bench"
        generator = paddle.Generator().manual_seed(0)
        output = pipe(prompt=prompt, image=image, generator=generator, output_type="np")
        image = output.images[0]
        assert image.shape == (512, 512, 3)
        image = image[-3:, -3:, -1]
        expected_image = [
            [[0.17348257], [0.15836588], [0.14607191]],
            [[0.17892927], [0.1668604], [0.15961224]],
            [[0.17489928], [0.1661663], [0.16446933]],
        ]
        assert np.abs(expected_image - image).max() < 0.05
