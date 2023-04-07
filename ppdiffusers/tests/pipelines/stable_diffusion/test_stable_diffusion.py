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
import tempfile
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

from ...models.test_models_unet_2d_condition import create_lora_layers


class StableDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.29159048, 0.20539099, 0.29126638, 0.19384867, 0.2436865, 0.45562512, 0.12645364, 0.14380667, 0.3520335]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_lora(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        lora_attn_procs = create_lora_layers(sd_pipe.unet)
        sd_pipe.unet.set_attn_processor(lora_attn_procs)
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.0})
        image = output.images
        image_slice_1 = image[0, -3:, -3:, -1]
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.5})
        image = output.images
        image_slice_2 = image[0, -3:, -3:, -1]
        assert np.abs(image_slice - image_slice_1).max() < 0.01
        assert np.abs(image_slice - image_slice_2).max() > 0.01

    def test_stable_diffusion_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = 3 * [inputs["prompt"]]
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]
        inputs = self.get_dummy_inputs()
        prompt = 3 * [inputs.pop("prompt")]
        text_inputs = sd_pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=sd_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
        )
        text_inputs = text_inputs["input_ids"]
        prompt_embeds = sd_pipe.text_encoder(text_inputs)[0]
        inputs["prompt_embeds"] = prompt_embeds
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 0.0001

    def test_stable_diffusion_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]
        inputs = self.get_dummy_inputs()
        prompt = 3 * [inputs.pop("prompt")]
        embeds = []
        for p in [prompt, negative_prompt]:
            text_inputs = sd_pipe.tokenizer(
                p,
                padding="max_length",
                max_length=sd_pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            )
            text_inputs = text_inputs["input_ids"]
            embeds.append(sd_pipe.text_encoder(text_inputs)[0])
        inputs["prompt_embeds"], inputs["negative_prompt_embeds"] = embeds
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 0.0001

    def test_stable_diffusion_ddim_factor_8(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs, height=136, width=136)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 136, 136, 3)
        expected_slice = np.array(
            [0.27389365, 0.76076514, 0.48075324, 0.6340331, 0.4487665, 0.4152641, 0.3356102, 0.3121925, 0.3853925]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_pndm(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = PNDMScheduler(skip_prk_steps=True)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.17811695, 0.21427456, 0.29789412, 0.21849585, 0.24531782, 0.4673458, 0.17078575, 0.13884068, 0.33425587]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_no_safety_checker(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-lms-pipe", safety_checker=None
        )
        assert isinstance(pipe, StableDiffusionPipeline)
        assert isinstance(pipe.scheduler, LMSDiscreteScheduler)
        assert pipe.safety_checker is None
        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = StableDiffusionPipeline.from_pretrained(tmpdirname, from_diffusers=False)
        assert pipe.safety_checker is None
        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None

    def test_stable_diffusion_k_lms(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.33460706, 0.19120479, 0.28665113, 0.18987545, 0.23559365, 0.4537152, 0.1421108, 0.1342963, 0.3254879]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_k_euler_ancestral(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.33486712, 0.19070494, 0.2863517, 0.19030812, 0.23497805, 0.45370454, 0.14277357, 0.13416818, 0.32546803]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_k_euler(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.33460736, 0.19120452, 0.2866511, 0.18987533, 0.2355937, 0.4537152, 0.14211062, 0.13429633, 0.3254879]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_vae_slicing(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        image_count = 4
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_1 = sd_pipe(**inputs)
        sd_pipe.enable_vae_slicing()
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_2 = sd_pipe(**inputs)
        assert np.abs(output_2.images.flatten() - output_1.images.flatten()).max() < 0.003

    def test_stable_diffusion_vae_tiling(self):
        components = self.get_dummy_components()

        # make sure here that pndm scheduler skips prk
        components["safety_checker"] = None
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test that tiled decode at 512x512 yields the same result as the non-tiled decode
        generator = paddle.Generator().manual_seed(0)
        output_1 = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")

        # make sure tiled vae decode yields the same result
        sd_pipe.enable_vae_tiling()
        generator = paddle.Generator().manual_seed(0)
        output_2 = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")

        assert np.abs(output_2.images.flatten() - output_1.images.flatten()).max() < 5e-1

    def test_stable_diffusion_negative_prompt(self):
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.26844203, 0.29748845, 0.290293, 0.30656427, 0.27700335, 0.48669702, 0.24277791, 0.21004745, 0.37346232]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_num_images_per_prompt(self):
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        images = sd_pipe(prompt, num_inference_steps=2, output_type="np").images
        assert images.shape == (1, 64, 64, 3)
        batch_size = 2
        images = sd_pipe([prompt] * batch_size, num_inference_steps=2, output_type="np").images
        assert images.shape == (batch_size, 64, 64, 3)
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt, num_inference_steps=2, output_type="np", num_images_per_prompt=num_images_per_prompt
        ).images
        assert images.shape == (num_images_per_prompt, 64, 64, 3)
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size, num_inference_steps=2, output_type="np", num_images_per_prompt=num_images_per_prompt
        ).images
        assert images.shape == (batch_size * num_images_per_prompt, 64, 64, 3)

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

    def test_stable_diffusion_height_width_opt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = "hey"
        output = sd_pipe(prompt, num_inference_steps=1, output_type="np")
        image_shape = output.images[0].shape[:2]
        assert image_shape == (64, 64)
        output = sd_pipe(prompt, num_inference_steps=1, height=96, width=96, output_type="np")
        image_shape = output.images[0].shape[:2]
        assert image_shape == (96, 96)
        config = dict(sd_pipe.unet.config)
        config["sample_size"] = 96
        sd_pipe.unet = UNet2DConditionModel.from_config(config)
        output = sd_pipe(prompt, num_inference_steps=1, output_type="np")
        image_shape = output.images[0].shape[:2]
        assert image_shape == (192, 192)


@slow
@require_paddle_gpu
class StableDiffusionPipelineSlowTests(unittest.TestCase):
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

    def test_stable_diffusion_1_1_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1")
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.43625, 0.43554, 0.3667, 0.4066, 0.39703, 0.38658, 0.43936, 0.43557, 0.40592])
        assert np.abs(image_slice - expected_slice).max() < 0.0001

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.574, 0.47841, 0.31625, 0.63583, 0.58306, 0.55056, 0.50825, 0.56306, 0.55748])
        assert np.abs(image_slice - expected_slice).max() < 0.0001

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.38019, 0.28647, 0.27321, 0.40377, 0.3829, 0.35446, 0.39218, 0.38165, 0.42239])
        assert np.abs(image_slice - expected_slice).max() < 0.0001

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.10542, 0.0962, 0.07332, 0.09015, 0.09382, 0.07597, 0.08496, 0.07806, 0.06455])
        assert np.abs(image_slice - expected_slice).max() < 0.0001

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.03503, 0.03494, 0.01087, 0.03128, 0.02552, 0.00803, 0.00742, 0.00372, 0.0])
        assert np.abs(image_slice - expected_slice).max() < 0.0001

    # def test_stable_diffusion_attention_slicing(self):
    #     pipe = StableDiffusionPipeline.from_pretrained(
    #         'CompVis/stable-diffusion-v1-4', paddle_dtype=paddle.float16)
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing()
    #     inputs = self.get_inputs(dtype='float16')
    #     image_sliced = pipe(**inputs).images
    #     mem_bytes = paddle.device.cuda.memory_allocated()
    #     assert mem_bytes < 3.75 * 10 ** 9
    #     pipe.disable_attention_slicing()
    #     inputs = self.get_inputs(dtype='float16')
    #     image = pipe(**inputs).images
    #     mem_bytes = paddle.device.cuda.memory_allocated()
    #     assert mem_bytes > 3.75 * 10 ** 9
    #     assert np.abs(image_sliced - image).max() < 0.001

    # def test_stable_diffusion_vae_slicing(self):
    #     pipe = StableDiffusionPipeline.from_pretrained(
    #         'CompVis/stable-diffusion-v1-4', paddle_dtype=paddle.float16)
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing()
    #     pipe.enable_vae_slicing()
    #     inputs = self.get_inputs(dtype='float16')
    #     inputs['prompt'] = [inputs['prompt']] * 4
    #     inputs['latents'] = paddle.concat(x=[inputs['latents']] * 4)
    #     image_sliced = pipe(**inputs).images
    #     mem_bytes = paddle.device.cuda.memory_allocated()
    #     assert mem_bytes < 4000000000.0
    #     pipe.disable_vae_slicing()
    #     inputs = self.get_inputs(dtype='float16')
    #     inputs['prompt'] = [inputs['prompt']] * 4
    #     inputs['latents'] = paddle.concat(x=[inputs['latents']] * 4)
    #     image = pipe(**inputs).images
    #     mem_bytes = paddle.device.cuda.memory_allocated()
    #     assert mem_bytes > 4000000000.0
    #     assert np.abs(image_sliced - image).max() < 0.01

    def test_stable_diffusion_fp16_vs_autocast(self):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs(dtype="float16")
        image_fp16 = pipe(**inputs).images
        with paddle.amp.auto_cast(True, level="O2"):
            inputs = self.get_inputs()
            image_autocast = pipe(**inputs).images
        diff = np.abs(image_fp16.flatten() - image_autocast.flatten())
        assert diff.mean() < 0.1

    def test_stable_diffusion_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: paddle.Tensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([-0.5693, -0.3018, -0.9746, 0.0518, -0.877, 0.7559, -1.7402, 0.1022, 1.1582])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.1958, -0.2993, -1.0166, -0.5005, -0.481, 0.6162, -0.9492, 0.6621, 1.4492]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05

        callback_fn.has_been_called = False
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(dtype="float16")
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]


@nightly
@require_paddle_gpu
class StableDiffusionPipelineNightlyTests(unittest.TestCase):
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

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_text2img/stable_diffusion_1_4_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_stable_diffusion_1_5_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = np.array(
            [
                [0.7839468, 0.6564859, 0.48896512],
                [0.78088367, 0.6400461, 0.447728],
                [0.81458974, 0.67865074, 0.51496047],
            ]
        )
        max_diff = np.abs(expected_image - image[0][0:3]).max()
        assert max_diff < 0.001

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_text2img/stable_diffusion_1_4_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_text2img/stable_diffusion_1_4_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_stable_diffusion_euler(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = np.array(
            [
                [0.7907467, 0.69895816, 0.5911293],
                [0.7878128, 0.6815276, 0.55695873],
                [0.79491043, 0.69076216, 0.58900857],
            ]
        )
        max_diff = np.abs(expected_image - image[0][0:3]).max()
        assert max_diff < 0.001

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 25
        image = sd_pipe(**inputs).images[0]
        expected_image = np.array(
            [[0.8398815, 0.7510048, 0.6475117], [0.8548264, 0.75703114, 0.63529825], [0.8559129, 0.75676, 0.6597851]]
        )
        max_diff = np.abs(expected_image - image[0][0:3]).max()
        assert max_diff < 0.001
