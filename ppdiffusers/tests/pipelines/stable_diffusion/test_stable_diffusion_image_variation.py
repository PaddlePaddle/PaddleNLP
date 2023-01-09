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
from PIL import Image
from test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import (
    CLIPFeatureExtractor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)
from ppdiffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    StableDiffusionImageVariationPipeline,
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


class StableDiffusionImageVariationPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionImageVariationPipeline

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
        image_encoder_config = dict(
            vision_embed_dim=32,
            projection_dim=32,
            vision_mlp_ratio=1,
            vision_heads=4,
            vision_layers=5,
            image_resolution=32,
            vision_patch_size=4,
        )
        image_encoder_config = CLIPVisionConfig.from_dict(image_encoder_config)
        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)
        image_encoder.eval()
        feature_extractor = CLIPFeatureExtractor(crop_size=32)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
            "safety_checker": None,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().transpose([0, 2, 3, 1])[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_img_variation_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImageVariationPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.3151392638683319,
                0.11757123470306396,
                0.30373042821884155,
                0.20310524106025696,
                0.20141255855560303,
                0.5957024097442627,
                0.25098103284835815,
                0.23880955576896667,
                0.5245755910873413,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img_variation_multiple_images(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImageVariationPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        inputs["image"] = 2 * [inputs["image"]]
        output = sd_pipe(**inputs)

        image = output.images

        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 64, 64, 3)
        expected_slice = np.array(
            [
                0.6137708425521851,
                0.6509915590286255,
                0.6715413331985474,
                0.522708535194397,
                0.31863510608673096,
                0.46015962958335876,
                0.5901781916618347,
                0.3334598243236542,
                0.41584062576293945,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img_variation_num_images_per_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImageVariationPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        # test num_images_per_prompt=1 (default)
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs).images

        assert images.shape == (1, 64, 64, 3)

        # test num_images_per_prompt=1 (default) for batch of images
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs["image"] = batch_size * [inputs["image"]]
        images = sd_pipe(**inputs).images

        assert images.shape == (batch_size, 64, 64, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (num_images_per_prompt, 64, 64, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs["image"] = batch_size * [inputs["image"]]
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (batch_size * num_images_per_prompt, 64, 64, 3)


@slow
class StableDiffusionImageVariationPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_imgvar/input_image_vermeer.png")
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = paddle.to_tensor(latents, dtype=dtype)
        inputs = {
            "image": init_image,
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    # lambdalabs/sd-image-variations-diffusers
    def test_stable_diffusion_img_variation_pipeline_default(self):
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "fusing/sd-image-variations-diffusers", safety_checker=None
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [
                0.5717014670372009,
                0.47024625539779663,
                0.47462183237075806,
                0.6388776898384094,
                0.5250844359397888,
                0.500831663608551,
                0.638043999671936,
                0.5769134163856506,
                0.5223015546798706,
            ]
        )
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_img_variation_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: paddle.Tensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [
                        -0.15777504444122314,
                        0.2840809226036072,
                        -0.8084282875061035,
                        -0.11453461647033691,
                        -1.3060585260391235,
                        0.7711117267608643,
                        -2.119102954864502,
                        0.041628703474998474,
                        1.6307991743087769,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-3
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [
                        0.5763851404190063,
                        1.703276515007019,
                        1.1998151540756226,
                        -2.148696184158325,
                        -1.9190490245819092,
                        0.6978652477264404,
                        -0.695953369140625,
                        1.0288335084915161,
                        1.5673096179962158,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "fusing/sd-image-variations-diffusers",
            safety_checker=None,
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]


@nightly
class StableDiffusionImageVariationPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_imgvar/input_image_vermeer.png")
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = paddle.to_tensor(latents, dtype=dtype)
        inputs = {
            "image": init_image,
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_img_variation_pndm(self):
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained("fusing/sd-image-variations-diffusers")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_imgvar/lambdalabs_variations_pndm.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img_variation_dpm(self):
        sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained("fusing/sd-image-variations-diffusers")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 25
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_imgvar/lambdalabs_variations_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
