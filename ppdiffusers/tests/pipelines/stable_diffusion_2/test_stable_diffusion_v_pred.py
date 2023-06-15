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

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusion2VPredictionPipelineFastTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @property
    def dummy_cond_unet(self):
        paddle.seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            attention_head_dim=(2, 4),
            use_linear_projection=True,
        )
        return model

    @property
    def dummy_vae(self):
        paddle.seed(0)
        model = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
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
            projection_dim=64,
        )
        return CLIPTextModel(config).eval()

    def test_stable_diffusion_v_pred_ddim(self):
        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="v_prediction",
        )
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")
        image = output.images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.12384933,
                0.19702056,
                0.25682122,
                0.29907784,
                0.18888032,
                0.40307283,
                0.28899065,
                0.21834826,
                0.41601387,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_v_pred_k_euler(self):
        unet = self.dummy_cond_unet
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", prediction_type="v_prediction"
        )
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")
        image = output.images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.1817798, 0.16936094, 0.18231615, 0.33563924, 0.17667511, 0.34496182, 0.45114157, 0.37192938, 0.45209426]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_v_pred_fp16(self):
        """Test that stable diffusion v-prediction works with fp16"""
        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="v_prediction",
        )
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        unet = unet.to(dtype=paddle.float16)
        vae = vae.to(dtype=paddle.float16)
        bert = bert.to(dtype=paddle.float16)
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        image = sd_pipe([prompt], generator=generator, num_inference_steps=2, output_type="np").images
        assert image.shape == (1, 64, 64, 3)


@slow
@require_paddle_gpu
class StableDiffusion2VPredictionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_v_pred_default(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
        sd_pipe.enable_attention_slicing()
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=7.5, num_inference_steps=20, output_type="np")
        image = output.images
        image_slice = image[0, 253:256, 253:256, -1]
        assert image.shape == (1, 768, 768, 3)
        expected_slice = np.array(
            [0.05667132, 0.05700234, 0.04156408, 0.04631725, 0.04327643, 0.06003231, 0.05165312, 0.05258191, 0.0865913]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_v_pred_upcast_attention(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", paddle_dtype=paddle.float16
        )
        sd_pipe.enable_attention_slicing()
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=7.5, num_inference_steps=20, output_type="np")
        image = output.images
        image_slice = image[0, 253:256, 253:256, -1]
        assert image.shape == (1, 768, 768, 3)

        expected_slice = np.array(
            [0.04541016, 0.04516602, 0.05493164, 0.05078125, 0.04296875, 0.07275391, 0.06567383, 0.0534668, 0.04833984]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05

    def test_stable_diffusion_v_pred_euler(self):
        scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2", subfolder="scheduler")
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", scheduler=scheduler)
        sd_pipe.enable_attention_slicing()
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt], generator=generator, num_inference_steps=5, output_type="numpy")
        image = output.images
        image_slice = image[0, 253:256, 253:256, -1]
        assert image.shape == (1, 768, 768, 3)
        expected_slice = np.array(
            [
                0.03515199,
                0.03756374,
                0.05046153,
                0.04240236,
                0.05509549,
                0.06556576,
                0.04710263,
                0.02758819,
                0.05959105,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_v_pred_dpm(self):
        """
        TODO: update this test after making DPM compatible with V-prediction!
        """
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2", subfolder="scheduler"
        )
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", scheduler=scheduler)
        sd_pipe.enable_attention_slicing()
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "a photograph of an astronaut riding a horse"
        generator = paddle.Generator().manual_seed(0)
        image = sd_pipe(
            [prompt], generator=generator, guidance_scale=7.5, num_inference_steps=5, output_type="numpy"
        ).images
        image_slice = image[0, 253:256, 253:256, -1]
        assert image.shape == (1, 768, 768, 3)
        expected_slice = np.array(
            [0.20492354, 0.2115368, 0.2323401, 0.2415919, 0.25598443, 0.24843931, 0.25171167, 0.23580211, 0.23604062]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    # def test_stable_diffusion_attention_slicing_v_pred(self):
    #     model_id = 'stabilityai/stable-diffusion-2'
    #     pipe = StableDiffusionPipeline.from_pretrained(model_id,
    #         paddle_dtype=paddle.float16)
    #     pipe.set_progress_bar_config(disable=None)
    #     prompt = 'a photograph of an astronaut riding a horse'
    #     pipe.enable_attention_slicing()
    #     generator = paddle.Generator().manual_seed(0)
    #     output_chunked = pipe([prompt], generator=generator, guidance_scale
    #         =7.5, num_inference_steps=10, output_type='numpy')
    #     image_chunked = output_chunked.images
    #     mem_bytes = paddle.device.cuda.memory_allocated()
    #     assert mem_bytes < 5.5 * 10 ** 9
    #     pipe.disable_attention_slicing()
    #     generator = paddle.Generator().manual_seed(0)
    #     output = pipe([prompt], generator=generator, guidance_scale=7.5,
    #         num_inference_steps=10, output_type='numpy')
    #     image = output.images
    #     mem_bytes = paddle.device.cuda.memory_allocated()
    #     assert mem_bytes > 5.5 * 10 ** 9
    #     assert np.abs(image_chunked.flatten() - image.flatten()).max() < 0.001

    def test_stable_diffusion_text2img_pipeline_v_pred_default(self):
        # invalid expected_image
        # expected_image = load_numpy(
        #     'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-text2img/astronaut_riding_a_horse_v_pred.npy'
        #     )
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=None)
        prompt = "astronaut riding a horse"
        generator = paddle.Generator().manual_seed(0)
        output = pipe(prompt=prompt, guidance_scale=7.5, generator=generator, output_type="np")
        image = output.images[0]
        assert image.shape == (768, 768, 3)
        expected_image = np.array(
            [0.26713198, 0.2630347, 0.25486767, 0.23375505, 0.24399692, 0.22363415, 0.24688962, 0.21346492, 0.23014635]
        )
        image = image[-3:, -3:, -1].flatten()
        assert np.abs(expected_image - image).max() < 0.075

    def test_stable_diffusion_text2img_pipeline_v_pred_fp16(self):
        # invalid expected_image
        # expected_image = load_numpy(
        #     'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-text2img/astronaut_riding_a_horse_v_pred_fp16.npy'
        #     )
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        prompt = "astronaut riding a horse"
        generator = paddle.Generator().manual_seed(0)
        output = pipe(prompt=prompt, guidance_scale=7.5, generator=generator, output_type="np")
        image = output.images[0]
        assert image.shape == (768, 768, 3)
        expected_image = np.array(
            [0.26220703, 0.25195312, 0.2434082, 0.22753906, 0.23632812, 0.21777344, 0.23901367, 0.20629883, 0.22192383]
        )
        image = image[-3:, -3:, -1].flatten()
        assert np.abs(expected_image - image).max() < 0.75

    def test_stable_diffusion_text2img_intermediate_state_v_pred(self):
        number_of_steps = 0

        def test_callback_fn(step: int, timestep: int, latents: paddle.Tensor) -> None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 96, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([-0.2542, -1.276, 0.426, -0.956, -1.173, -0.5884, 2.416, 0.1553, -1.21])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
            elif step == 19:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 96, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.959, -0.964, -0.614, 0.0977, -0.6953, -0.2343, 1.551, -0.03357, -0.11395]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05

        test_callback_fn.has_been_called = False
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        prompt = "Andromeda galaxy in a bottle"
        generator = paddle.Generator().manual_seed(0)
        pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            callback=test_callback_fn,
            callback_steps=1,
        )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 20
