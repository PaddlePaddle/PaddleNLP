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
import random
import unittest

import numpy as np
import paddle
from test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import (
    TEST_DOWNLOAD_SERVER,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    slow,
)


class StableDiffusionImg2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionImg2ImgPipeline

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
        scheduler = PNDMScheduler(skip_prk_steps=True)

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
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_img2img_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [
                0.6644147038459778,
                0.4746779501438141,
                0.40329617261886597,
                0.5751277208328247,
                0.5119695663452148,
                0.5045669674873352,
                0.45643243193626404,
                0.4374881982803345,
                0.5445119738578796,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_negative_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [
                0.7663768529891968,
                0.42991745471954346,
                0.4016766846179962,
                0.6222720742225647,
                0.4062548279762268,
                0.4925062954425812,
                0.4996223449707031,
                0.44043585658073425,
                0.5554477572441101,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_multiple_images(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = inputs["image"].tile([2, 1, 1, 1])
        image = sd_pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 32, 32, 3)
        expected_slice = np.array(
            [
                0.4021262228488922,
                0.19536930322647095,
                0.3976094126701355,
                0.7523422241210938,
                0.33938348293304443,
                0.47887665033340454,
                0.6323817372322083,
                0.40313106775283813,
                0.47351086139678955,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_k_lms(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [
                0.055213481187820435,
                0.7224215269088745,
                0.6463539004325867,
                0.21371740102767944,
                0.8918260335922241,
                0.49397602677345276,
                0.4064573049545288,
                0.6090186834335327,
                0.45809662342071533,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_num_images_per_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        # test num_images_per_prompt=1 (default)
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs).images

        assert images.shape == (1, 32, 32, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * batch_size
        images = sd_pipe(**inputs).images

        assert images.shape == (batch_size, 32, 32, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (num_images_per_prompt, 32, 32, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * batch_size
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)


@slow
class StableDiffusionImg2ImgPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_img2img/sketch-mountains-input.png")
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_img2img_default(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.27150, 0.14849, 0.15605, 0.26740, 0.16954, 0.18204, 0.31470, 0.26311, 0.24525])
        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_k_lms(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.04890, 0.04862, 0.06422, 0.04655, 0.05108, 0.05307, 0.05926, 0.08759, 0.06852])
        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_ddim(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.06069, 0.05703, 0.08054, 0.05797, 0.06286, 0.06234, 0.08438, 0.11151, 0.08068])
        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: paddle.Tensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [
                        0.7650054097175598,
                        0.10256098955869675,
                        0.4976114332675934,
                        3.388350009918213,
                        3.7242040634155273,
                        4.272988796234131,
                        2.4656283855438232,
                        3.483647108078003,
                        1.765011191368103,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [
                        0.7580092549324036,
                        0.10288780182600021,
                        0.4941849708557129,
                        3.3663346767425537,
                        3.7071609497070312,
                        4.25173807144165,
                        2.4461638927459717,
                        3.451681137084961,
                        1.761878490447998,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3

        callback_fn.has_been_called = False

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            safety_checker=None,
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 2


@nightly
class StableDiffusionImg2ImgPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_img2img/sketch-mountains-input.png")
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 50,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_img2img_pndm(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_img2img/stable_diffusion_1_5_pndm.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_ddim(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_img2img/stable_diffusion_1_5_ddim.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_lms(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_img2img/stable_diffusion_1_5_lms.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_dpm(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 30
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_img2img/stable_diffusion_1_5_dpm.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
