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
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)
from ppdiffusers.utils import load_numpy, nightly, slow
from ppdiffusers.utils.testing_utils import CaptureLogger, require_paddle_gpu


class StableDiffusion2PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS

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
            attention_head_dim=(2, 4),
            use_linear_projection=True,
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
            sample_size=128,
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
            hidden_act="gelu",
            projection_dim=512,
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
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_ddim(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5649, 0.6022, 0.4804, 0.527, 0.5585, 0.4643, 0.5159, 0.4963, 0.4793])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_pndm(self):
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.13812214, 0.2526546, 0.36139387, 0.2045241, 0.28851387, 0.48496005, 0.17970857, 0.17119485, 0.349154]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_k_lms(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.19994166, 0.15929759, 0.34467825, 0.26391482, 0.22975561, 0.4496565, 0.23415366, 0.16101378, 0.33743745]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_k_euler_ancestral(self):
        components = self.get_dummy_components()
        components["scheduler"] = EulerAncestralDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4715, 0.5376, 0.4569, 0.5224, 0.5734, 0.4797, 0.5465, 0.5074, 0.5046])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_k_euler(self):
        components = self.get_dummy_components()
        components["scheduler"] = EulerDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4717, 0.5376, 0.4568, 0.5225, 0.5734, 0.4797, 0.5467, 0.5074, 0.5043])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_long_prompt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        do_classifier_free_guidance = True
        negative_prompt = None
        num_images_per_prompt = 1
        logger = logging.get_logger("ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion")
        prompt = 25 * "@"
        with CaptureLogger(logger) as cap_logger_3:
            text_embeddings_3 = sd_pipe._encode_prompt(
                prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        prompt = 100 * "@"
        with CaptureLogger(logger) as cap_logger:
            text_embeddings = sd_pipe._encode_prompt(
                prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        negative_prompt = "Hello"
        with CaptureLogger(logger) as cap_logger_2:
            text_embeddings_2 = sd_pipe._encode_prompt(
                prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        assert text_embeddings_3.shape == text_embeddings_2.shape == text_embeddings.shape
        assert text_embeddings.shape[1] == 77
        assert cap_logger.out == cap_logger_2.out
        assert cap_logger.out.count("@") == 25
        assert cap_logger_3.out == ""


@slow
@require_paddle_gpu
class StableDiffusion2PipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = paddle.to_tensor(latents).cast(dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_default_ddim(self):
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.49493, 0.47896, 0.40798, 0.54214, 0.53212, 0.48202, 0.47656, 0.46329, 0.48506])
        assert np.abs(image_slice - expected_slice).max() < 0.0001

    def test_stable_diffusion_pndm(self):
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.49493, 0.47896, 0.40798, 0.54214, 0.53212, 0.48202, 0.47656, 0.46329, 0.48506])
        assert np.abs(image_slice - expected_slice).max() < 0.0001

    def test_stable_diffusion_k_lms(self):
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.1044, 0.13115, 0.111, 0.10141, 0.1144, 0.07215, 0.11332, 0.09693, 0.10006])
        assert np.abs(image_slice - expected_slice).max() < 0.0001

    # def test_stable_diffusion_attention_slicing(self):
    #     pipe = StableDiffusionPipeline.from_pretrained(
    #         "stabilityai/stable-diffusion-2-base", paddle_dtype=paddle.float16
    #     )
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing()
    #     inputs = self.get_inputs(dtype="float16")
    #     image_sliced = pipe(**inputs).images
    #     mem_bytes = paddle.device.cuda.memory_allocated()
    #     assert mem_bytes < 3.3 * 10**9
    #     pipe.disable_attention_slicing()
    #     inputs = self.get_inputs(dtype="float16")
    #     image = pipe(**inputs).images
    #     mem_bytes = paddle.device.cuda.memory_allocated()
    #     assert mem_bytes > 3.3 * 10**9
    #     assert np.abs(image_sliced - image).max() < 0.001

    def test_stable_diffusion_text2img_intermediate_state(self):
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
                    [-0.3862, -0.4507, -1.1729, 0.0686, -1.1045, 0.7124, -1.8301, 0.1903, 1.2773]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.272, -0.1863, -0.7383, -0.5029, -0.7534, 0.397, -0.7646, 0.4468, 1.2686])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05

        callback_fn.has_been_called = False
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base", paddle_dtype=paddle.float16
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(dtype="float16")
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]


@nightly
@require_paddle_gpu
class StableDiffusion2PipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = paddle.to_tensor(latents).cast(dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_2_0_default_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_0_base_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.01

    def test_stable_diffusion_2_1_default_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.01

    def test_stable_diffusion_ddim(self):  # not pass
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.01

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.01

    def test_stable_diffusion_euler(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_euler.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.01

    def test_stable_diffusion_dpm(self):  # not pass
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 25
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.01
