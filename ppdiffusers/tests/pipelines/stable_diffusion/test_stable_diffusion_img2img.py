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
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, nightly, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusionImg2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
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
            [0.65765685, 0.33105016, 0.30553353, 0.62215245, 0.3664353, 0.56206244, 0.43910763, 0.41746795, 0.5354696]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

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
            [0.80471164, 0.46412903, 0.37454736, 0.4387825, 0.3394906, 0.5412611, 0.4721696, 0.49021164, 0.5914886]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_stable_diffusion_img2img_multiple_init_images(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = inputs["image"].tile(repeat_times=[2, 1, 1, 1])
        image = sd_pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]
        assert image.shape == (2, 32, 32, 3)
        expected_slice = np.array(
            [0.25047868, 0.24038231, 0.40680486, 0.52044916, 0.08695492, 0.27611375, 0.5864357, 0.03654012, 0.24583888]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

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
            [0.0489203, 0.729686, 0.6543164, 0.1997531, 0.90586334, 0.51476943, 0.3933796, 0.620923, 0.45674747]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_stable_diffusion_img2img_num_images_per_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs).images
        assert images.shape == (1, 32, 32, 3)
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * batch_size
        images = sd_pipe(**inputs).images
        assert images.shape == (batch_size, 32, 32, 3)
        num_images_per_prompt = 2
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images
        assert images.shape == (num_images_per_prompt, 32, 32, 3)
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * batch_size
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images
        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)


@slow
@require_paddle_gpu
class StableDiffusionImg2ImgPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/sketch-mountains-input.png"
        )
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
        assert np.abs(expected_slice - image_slice).max() < 0.001

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
        assert np.abs(expected_slice - image_slice).max() < 0.001

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
        assert np.abs(expected_slice - image_slice).max() < 0.001

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
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
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
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05

        callback_fn.has_been_called = False
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, paddle_dtype=paddle.float16
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(dtype="float16")
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 2

    def test_stable_diffusion_img2img_pipeline_multiple_of_8(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((760, 504))
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        prompt = "A fantasy landscape, trending on artstation"
        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, generator=generator, output_type="np"
        )
        image = output.images[0]
        image_slice = image[255:258, 383:386, -1]
        assert image.shape == (504, 760, 3)
        expected_slice = np.array(
            [0.71240354, 0.71053374, 0.69922864, 0.7139934, 0.7106118, 0.69451976, 0.71982634, 0.71717453, 0.70306426]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005


@nightly
@require_paddle_gpu
class StableDiffusionImg2ImgPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/sketch-mountains-input.png"
        )
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
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/stable_diffusion_1_5_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_img2img_ddim(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/stable_diffusion_1_5_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_img2img_lms(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/stable_diffusion_1_5_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_img2img_dpm(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 30
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/stable_diffusion_1_5_dpm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001
