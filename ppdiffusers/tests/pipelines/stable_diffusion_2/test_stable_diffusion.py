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
from ppdiffusers.utils.testing_utils import CaptureLogger


class StableDiffusion2PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionPipeline

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
            # SD2-specific config below
            attention_head_dim=(2, 4, 8, 8),
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
        text_encoder_config = dict(
            text_embed_dim=32,
            text_heads=4,
            text_layers=5,
            vocab_size=1000,
            # SD2-specific config below
            text_hidden_act="gelu",
            projection_dim=512,
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
        expected_slice = np.array(
            [
                0.2780013084411621,
                0.300104022026062,
                0.39301803708076477,
                0.16988825798034668,
                0.28341245651245117,
                0.4687179923057556,
                0.1901910901069641,
                0.24267145991325378,
                0.40805837512016296,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

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
            [
                0.16097530722618103,
                0.2921574115753174,
                0.3814927339553833,
                0.10686677694320679,
                0.3050421476364136,
                0.49412596225738525,
                0.1495760679244995,
                0.23741552233695984,
                0.3981768488883972,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

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
            [
                0.30069679021835327,
                0.2993624210357666,
                0.40976178646087646,
                0.16049659252166748,
                0.28706830739974976,
                0.479509174823761,
                0.1975727081298828,
                0.2516917884349823,
                0.3934157192707062,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_k_euler_ancestral(self):
        components = self.get_dummy_components()
        components["scheduler"] = EulerAncestralDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.3009917140007019,
                0.2990441918373108,
                0.40941137075424194,
                0.16082444787025452,
                0.28649193048477173,
                0.4792911410331726,
                0.19813573360443115,
                0.2516951560974121,
                0.3934096097946167,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_k_euler(self):
        components = self.get_dummy_components()
        components["scheduler"] = EulerDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.3006975054740906,
                0.29936230182647705,
                0.40976184606552124,
                0.16049709916114807,
                0.28706830739974976,
                0.4795089662075043,
                0.19757294654846191,
                0.25169193744659424,
                0.3934159278869629,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

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
        # 100 - 77 + 1 (BOS token) + 1 (EOS token) = 25
        assert cap_logger.out.count("@") == 25
        assert cap_logger_3.out == ""


@slow
class StableDiffusion2PipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = paddle.to_tensor(latents, dtype=dtype)
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
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_pndm(self):
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.49493, 0.47896, 0.40798, 0.54214, 0.53212, 0.48202, 0.47656, 0.46329, 0.48506])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_k_lms(self):
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.10440, 0.13115, 0.11100, 0.10141, 0.11440, 0.07215, 0.11332, 0.09693, 0.10006])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

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
                    [
                        -0.3864480257034302,
                        -0.4536583125591278,
                        -1.1725395917892456,
                        0.06786295771598816,
                        -1.1069486141204834,
                        0.7110416293144226,
                        -1.8251583576202393,
                        0.19316525757312775,
                        1.2804031372070312,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-3
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [
                        0.26776331663131714,
                        -0.22403457760810852,
                        -0.8066927790641785,
                        -0.5501354932785034,
                        -0.8206711411476135,
                        0.39952874183654785,
                        -0.7856663465499878,
                        0.482083797454834,
                        1.2937829494476318,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base",
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]


@nightly
class StableDiffusion2PipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = paddle.to_tensor(latents, dtype=dtype)
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
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
            "/stable_diffusion_2_text2img/stable_diffusion_2_0_base_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_2_1_default_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
            "/stable_diffusion_2_text2img/stable_diffusion_2_1_base_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
            "/stable_diffusion_2_text2img/stable_diffusion_2_1_base_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
            "/stable_diffusion_2_text2img/stable_diffusion_2_1_base_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_euler(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
            "/stable_diffusion_2_text2img/stable_diffusion_2_1_base_euler.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 25
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
            "/stable_diffusion_2_text2img/stable_diffusion_2_1_base_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
