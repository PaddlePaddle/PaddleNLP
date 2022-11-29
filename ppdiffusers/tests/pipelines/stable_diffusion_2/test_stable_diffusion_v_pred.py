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

from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import load_numpy, slow


class StableDiffusion2VPredictionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
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
            # SD2-specific config below
            attention_head_dim=(2, 4, 8, 8),
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
        config = dict(
            text_embed_dim=32,
            text_heads=4,
            text_layers=5,
            vocab_size=1000,
            # SD2-specific config below
            text_hidden_act="gelu",
            projection_dim=64,
        )
        model = CLIPTextModel(**config)
        model.eval()
        return model

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

        # make sure here that pndm scheduler skips prk
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
                0.30307483673095703,
                0.3711841404438019,
                0.2971755266189575,
                0.21124267578125,
                0.24938753247261047,
                0.4050242006778717,
                0.22494885325431824,
                0.21655383706092834,
                0.42639416456222534,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_v_pred_k_euler(self):
        unet = self.dummy_cond_unet
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", prediction_type="v_prediction"
        )
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
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
                0.28467634320259094,
                0.3477485179901123,
                0.2706109285354614,
                0.2010326385498047,
                0.24505868554115295,
                0.36301904916763306,
                0.3366130590438843,
                0.32769495248794556,
                0.44537732005119324,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
class StableDiffusion2VPredictionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
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
        expected_slice = np.array([0.0567, 0.057, 0.0416, 0.0463, 0.0433, 0.06, 0.0517, 0.0526, 0.0866])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

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
        expected_slice = np.array([0.0351, 0.0376, 0.0505, 0.0424, 0.0551, 0.0656, 0.0471, 0.0276, 0.0596])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

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
            [
                0.2049230933189392,
                0.21153515577316284,
                0.2323405146598816,
                0.24159285426139832,
                0.2559834122657776,
                0.2484399378299713,
                0.25171154737472534,
                0.2358025312423706,
                0.23604032397270203,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_text2img_pipeline_v_pred_default(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/"
            "sd2-text2img/astronaut_riding_a_horse_v_pred.npy"
        )

        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=None)

        prompt = "astronaut riding a horse"

        generator = paddle.Generator().manual_seed(0)
        output = pipe(prompt=prompt, guidance_scale=7.5, generator=generator, output_type="np")
        image = output.images[0]

        assert image.shape == (768, 768, 3)
        assert np.abs(expected_image - image).max() < 5e-3

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
                expected_slice = np.array(
                    [-0.2543, -1.2755, 0.4261, -0.9555, -1.173, -0.5892, 2.4159, 0.1554, -1.2098]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-3
            elif step == 19:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 96, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.9572, -0.967, -0.6152, 0.0894, -0.699, -0.2344, 1.5465, -0.0357, -0.1141]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-2

        test_callback_fn.has_been_called = False

        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
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
