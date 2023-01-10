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
from ppdiffusers.utils import TEST_DOWNLOAD_SERVER, load_numpy, nightly, slow
from ppdiffusers.utils.testing_utils import CaptureLogger


class StableDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
        config = dict(
            text_embed_dim=32,
            text_heads=4,
            text_layers=5,
            vocab_size=1000,
        )
        config = CLIPTextConfig.from_dict(config)
        text_encoder = CLIPTextModel(config)
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
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.23783081769943237,
                0.2683892548084259,
                0.3141400218009949,
                0.2046811282634735,
                0.35299813747406006,
                0.5098875761032104,
                0.1809311807155609,
                0.22014355659484863,
                0.37353450059890747,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

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
            [
                0.3770627975463867,
                0.8549360632896423,
                0.5036248564720154,
                0.7778869271278381,
                0.5413415431976318,
                0.43612781167030334,
                0.3498867154121399,
                0.3073400855064392,
                0.3994869291782379,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

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
            [
                0.20226451754570007,
                0.2968420088291168,
                0.33756476640701294,
                0.2581881880760193,
                0.32979750633239746,
                0.5054152607917786,
                0.21498948335647583,
                0.1985326111316681,
                0.36852598190307617,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    # def test_stable_diffusion_no_safety_checker(self):
    #     pipe = StableDiffusionPipeline.from_pretrained(
    #         "hf-internal-testing/tiny-stable-diffusion-lms-pipe",
    #         safety_checker=None)
    #     assert isinstance(pipe, StableDiffusionPipeline)
    #     assert isinstance(pipe.scheduler, LMSDiscreteScheduler)
    #     assert pipe.safety_checker is None

    #     image = pipe("example prompt", num_inference_steps=2).images[0]
    #     assert image is not None

    #     # check that there's no error when saving a pipeline with one of the models being None
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         pipe.save_pretrained(tmpdirname)
    #         pipe = StableDiffusionPipeline.from_pretrained(tmpdirname)

    #     # sanity check that the pipeline still works
    #     assert pipe.safety_checker is None
    #     image = pipe("example prompt", num_inference_steps=2).images[0]
    #     assert image is not None

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
            [
                0.26912739872932434,
                0.2649509310722351,
                0.30547958612442017,
                0.1875210702419281,
                0.3605213165283203,
                0.5184987187385559,
                0.18579843640327454,
                0.22257044911384583,
                0.3584556579589844,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

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
            [
                0.2693905532360077,
                0.26448899507522583,
                0.3050641417503357,
                0.18792355060577393,
                0.36001259088516235,
                0.5184952020645142,
                0.1864156723022461,
                0.22257667779922485,
                0.3584723174571991,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

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
            [
                0.26912766695022583,
                0.2649509310722351,
                0.30547958612442017,
                0.18752089142799377,
                0.3605214059352875,
                0.5184987783432007,
                0.18579840660095215,
                0.22257041931152344,
                0.35845568776130676,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    # def test_stable_diffusion_vae_slicing(self):
    #     components = self.get_dummy_components()
    #     components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
    #     sd_pipe = StableDiffusionPipeline(**components)
    #     sd_pipe.set_progress_bar_config(disable=None)

    #     image_count = 4

    #     inputs = self.get_dummy_inputs()
    #     inputs["prompt"] = [inputs["prompt"]] * image_count
    #     output_1 = sd_pipe(**inputs)

    #     # make sure sliced vae decode yields the same result
    #     sd_pipe.enable_vae_slicing()
    #     inputs = self.get_dummy_inputs()
    #     inputs["prompt"] = [inputs["prompt"]] * image_count
    #     output_2 = sd_pipe(**inputs)

    #     # there is a small discrepancy at image borders vs. full batch decode
    #     assert np.abs(output_2.images.flatten() - output_1.images.flatten()).max() < 3e-3

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
            [
                0.18583709001541138,
                0.2710910439491272,
                0.33035263419151306,
                0.28418004512786865,
                0.31744185090065,
                0.5189821124076843,
                0.23001152276992798,
                0.20550400018692017,
                0.38687920570373535,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_num_images_per_prompt(self):
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # test num_images_per_prompt=1 (default)
        images = sd_pipe(prompt, num_inference_steps=2, output_type="np").images

        assert images.shape == (1, 64, 64, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        images = sd_pipe([prompt] * batch_size, num_inference_steps=2, output_type="np").images

        assert images.shape == (batch_size, 64, 64, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt, num_inference_steps=2, output_type="np", num_images_per_prompt=num_images_per_prompt
        ).images

        assert images.shape == (num_images_per_prompt, 64, 64, 3)

        # test num_images_per_prompt for batch of prompts
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
        # 100 - 77 + 1 (BOS token) + 1 (EOS token) = 25
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
class StableDiffusionPipelineSlowTests(unittest.TestCase):
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

    def test_stable_diffusion_1_1_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [
                0.5739878416061401,
                0.4783928692340851,
                0.3162396252155304,
                0.6358233690261841,
                0.5830500721931458,
                0.550546407699585,
                0.508239209651947,
                0.5630548000335693,
                0.5574693083763123,
            ]
        )
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.57400, 0.47841, 0.31625, 0.63583, 0.58306, 0.55056, 0.50825, 0.56306, 0.55748])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.38019, 0.28647, 0.27321, 0.40377, 0.38290, 0.35446, 0.39218, 0.38165, 0.42239])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [
                0.10543090105056763,
                0.09621396660804749,
                0.07332965731620789,
                0.0901607871055603,
                0.09383168816566467,
                0.07598823308944702,
                0.0849754810333252,
                0.07807475328445435,
                0.06456810235977173,
            ]
        )
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.03503, 0.03494, 0.01087, 0.03128, 0.02552, 0.00803, 0.00742, 0.00372, 0.00000])
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
                        -0.5656787157058716,
                        -0.3015173673629761,
                        -0.9714246988296509,
                        0.05416536331176758,
                        -0.8775155544281006,
                        0.7535261511802673,
                        -1.734954833984375,
                        0.10168607532978058,
                        1.1522550582885742,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [
                        -0.175542414188385,
                        -0.30413269996643066,
                        -1.0064715147018433,
                        -0.5015721321105957,
                        -0.4677184820175171,
                        0.601168692111969,
                        -0.9374254941940308,
                        0.6590529084205627,
                        1.446482539176941,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]


@nightly
class StableDiffusionPipelineNightlyTests(unittest.TestCase):
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

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_text2img/stable_diffusion_1_4_pndm.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_1_5_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_text2img/stable_diffusion_1_5_pndm.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_text2img/stable_diffusion_1_4_ddim.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_text2img/stable_diffusion_1_4_lms.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_euler(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_text2img/stable_diffusion_1_4_euler.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 25
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            f"{TEST_DOWNLOAD_SERVER}/stable_diffusion_text2img/stable_diffusion_1_4_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
