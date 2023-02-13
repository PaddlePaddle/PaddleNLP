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

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInpaintPipelineLegacy,
    UNet2DConditionModel,
    UNet2DModel,
    VQModel,
)
from ppdiffusers.utils import (
    TEST_DOWNLOAD_SERVER,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    slow,
)


class StableDiffusionInpaintLegacyPipelineFastTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
        return image

    @property
    def dummy_uncond_unet(self):
        paddle.seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

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
        )
        return model

    @property
    def dummy_cond_unet_inpaint(self):
        paddle.seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        return model

    @property
    def dummy_vq_model(self):
        paddle.seed(0)
        model = VQModel(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=3,
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
        )
        config = CLIPTextConfig.from_dict(config)
        model = CLIPTextModel(config)
        model.eval()
        return model

    @property
    def dummy_extractor(self):
        def extract(*args, **kwargs):
            class Out:
                def __init__(self):
                    self.pixel_values = paddle.ones([0])

                def to(self, device):
                    return self

            return Out()

        return extract

    def test_stable_diffusion_inpaint_legacy(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image_dummy = self.dummy_image.transpose([0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image_dummy)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image_dummy + 4)).convert("RGB").resize((32, 32))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipelineLegacy(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        )

        image = output.images

        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [
                0.0,
                0.40522676706314087,
                0.22000649571418762,
                0.3371007442474365,
                0.6130789518356323,
                0.5699742436408997,
                0.4295768141746521,
                0.6762629747390747,
                0.49702852964401245,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_legacy_negative_prompt(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image_dummy = self.dummy_image.transpose([0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image_dummy)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image_dummy + 4)).convert("RGB").resize((32, 32))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipelineLegacy(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        negative_prompt = "french fries"
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe(
            prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [
                0.0,
                0.41603270173072815,
                0.25802189111709595,
                0.2814731001853943,
                0.5799428224563599,
                0.5783349275588989,
                0.38746166229248047,
                0.6701520681381226,
                0.5042654275894165,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_legacy_num_images_per_prompt(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image_dummy = self.dummy_image.transpose([0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image_dummy)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image_dummy + 4)).convert("RGB").resize((32, 32))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipelineLegacy(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # test num_images_per_prompt=1 (default)
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        ).images

        assert images.shape == (1, 32, 32, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        ).images

        assert images.shape == (batch_size, 32, 32, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (num_images_per_prompt, 32, 32, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)


@slow
class StableDiffusionInpaintLegacyPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_inpaint/input_bench_image.png")
        mask_image = load_image(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_inpaint/input_bench_mask.png")
        inputs = {
            "prompt": "A red cat sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_inpaint_legacy_pndm(self):
        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.27200, 0.29103, 0.34405, 0.21418, 0.26317, 0.34281, 0.18033, 0.24911, 0.32028])
        assert np.abs(expected_slice - image_slice).max() < 1e-4

    def test_stable_diffusion_inpaint_legacy_k_lms(self):
        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.29014, 0.28882, 0.32835, 0.26502, 0.28182, 0.31162, 0.29297, 0.29534, 0.28214])
        assert np.abs(expected_slice - image_slice).max() < 1e-4

    def test_stable_diffusion_inpaint_legacy_intermediate_state(self):
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
                        -0.10290834307670593,
                        1.41594660282135,
                        -0.021856456995010376,
                        -0.5102450251579285,
                        -0.591240644454956,
                        0.195754736661911,
                        0.7505455017089844,
                        0.3472331464290619,
                        -1.3563168048858643,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [
                        0.47839266061782837,
                        1.1574846506118774,
                        0.6261610984802246,
                        0.2289661169052124,
                        0.25498080253601074,
                        -0.14365366101264954,
                        0.7087358832359314,
                        -0.1603042483329773,
                        -0.5652803778648376,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3

        callback_fn.has_been_called = False

        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
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
class StableDiffusionInpaintLegacyPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_inpaint/input_bench_image.png")
        mask_image = load_image(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_inpaint/input_bench_mask.png")
        inputs = {
            "prompt": "A red cat sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 50,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_inpaint_pndm(self):
        sd_pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_inpaint_legacy/stable_diffusion_1_5_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_ddim(self):
        sd_pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_inpaint_legacy/stable_diffusion_1_5_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_lms(self):
        sd_pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_inpaint_legacy/stable_diffusion_1_5_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_dpm(self):
        sd_pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 30
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_inpaint_legacy/stable_diffusion_1_5_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
