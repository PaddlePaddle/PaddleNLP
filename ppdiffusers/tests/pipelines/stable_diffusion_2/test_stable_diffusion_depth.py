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
import tempfile
import unittest

import numpy as np
import paddle
from PIL import Image
from test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    DPTConfig,
    DPTForDepthEstimation,
    DPTImageProcessor,
)
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionDepth2ImgPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, nightly, slow


class StableDiffusionDepth2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionDepth2ImgPipeline
    test_save_load_optional_components = False

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=5,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            attention_head_dim=(2, 4, 8, 8),
            use_linear_projection=True,
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

        backbone_config = {
            "global_padding": "same",
            "layer_type": "bottleneck",
            "depths": [3, 4, 9],
            "out_features": ["stage1", "stage2", "stage3"],
            "embedding_dynamic_padding": True,
            "hidden_sizes": [96, 192, 384, 768],
            "num_groups": 2,
            "return_dict": True,
        }
        depth_estimator_config = DPTConfig(
            image_size=32,
            patch_size=16,
            num_channels=3,
            hidden_size=32,
            num_hidden_layers=4,
            backbone_out_indices=(0, 1, 2, 3),
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
            initializer_range=0.02,
            is_hybrid=True,
            backbone_config=backbone_config,
            backbone_featmap_shape=[1, 384, 24, 24],
            return_dict=True,
        )
        depth_estimator = DPTForDepthEstimation(depth_estimator_config)
        depth_estimator.eval()
        feature_extractor = DPTImageProcessor.from_pretrained("hf-internal-testing/tiny-random-DPTForDepthEstimation")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "depth_estimator": depth_estimator,
            "feature_extractor": feature_extractor,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().transpose([0, 2, 3, 1])[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
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

    def test_save_load_local(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 1e-4)

    def test_dict_tuple_outputs_equivalent(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs())[0]
        output_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        max_diff = np.abs(output - output_tuple).max()
        self.assertLess(max_diff, 1e-4)

    def test_num_inference_steps_consistent(self):
        super().test_num_inference_steps_consistent()

    def test_progress_bar(self):
        super().test_progress_bar()

    def test_stable_diffusion_depth2img_default_case(self):
        components = self.get_dummy_components()
        pipe = StableDiffusionDepth2ImgPipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [
                0.36208441853523254,
                0.30812186002731323,
                0.30757439136505127,
                0.37698906660079956,
                0.34277722239494324,
                0.496349036693573,
                0.23618823289871216,
                0.3135001063346863,
                0.4720374345779419,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_depth2img_negative_prompt(self):
        components = self.get_dummy_components()
        pipe = StableDiffusionDepth2ImgPipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        negative_prompt = "french fries"
        output = pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [
                0.46177974343299866,
                0.4092407822608948,
                0.4314950704574585,
                0.4775702953338623,
                0.45297110080718994,
                0.5985510349273682,
                0.3381749987602234,
                0.4235011339187622,
                0.5393458604812622,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_depth2img_multiple_init_images(self):
        components = self.get_dummy_components()
        pipe = StableDiffusionDepth2ImgPipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = 2 * [inputs["image"]]
        image = pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 32, 32, 3)

        expected_slice = np.array(
            [
                0.49012109637260437,
                0.34430721402168274,
                0.3286625146865845,
                0.39489197731018066,
                0.14197629690170288,
                0.21738004684448242,
                0.23562008142471313,
                0.2237577736377716,
                0.35953062772750854,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_depth2img_num_images_per_prompt(self):
        components = self.get_dummy_components()
        pipe = StableDiffusionDepth2ImgPipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        # test num_images_per_prompt=1 (default)
        inputs = self.get_dummy_inputs()
        images = pipe(**inputs).images

        assert images.shape == (1, 32, 32, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * batch_size
        images = pipe(**inputs).images

        assert images.shape == (batch_size, 32, 32, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        inputs = self.get_dummy_inputs()
        images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (num_images_per_prompt, 32, 32, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * batch_size
        images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)

    def test_stable_diffusion_depth2img_pil(self):
        components = self.get_dummy_components()
        pipe = StableDiffusionDepth2ImgPipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()

        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        expected_slice = np.array(
            [
                0.36208441853523254,
                0.30812186002731323,
                0.30757439136505127,
                0.37698906660079956,
                0.34277722239494324,
                0.496349036693573,
                0.23618823289871216,
                0.3135001063346863,
                0.4720374345779419,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


@slow
class StableDiffusionDepth2ImgPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype=paddle.float32, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests/depth2img/two_cats.png"
        )
        inputs = {
            "prompt": "two tigers",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_depth2img_pipeline_default(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 480, 640, 3)
        expected_slice = np.array([0.75446, 0.74692, 0.75951, 0.81611, 0.80593, 0.79992, 0.90529, 0.87921, 0.86903])
        assert np.abs(expected_slice - image_slice).max() < 1e-4

    def test_stable_diffusion_depth2img_pipeline_k_lms(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to()
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 480, 640, 3)
        expected_slice = np.array([0.63957, 0.64879, 0.65668, 0.64385, 0.67078, 0.63588, 0.66577, 0.62180, 0.66286])
        assert np.abs(expected_slice - image_slice).max() < 1e-4

    def test_stable_diffusion_depth2img_pipeline_ddim(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to()
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 480, 640, 3)
        expected_slice = np.array([0.62840, 0.64191, 0.62953, 0.63653, 0.64205, 0.61574, 0.62252, 0.65827, 0.64809])
        assert np.abs(expected_slice - image_slice).max() < 1e-4

    def test_stable_diffusion_depth2img_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: paddle.Tensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 60, 80)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [
                        -1.1475517749786377,
                        -0.21485021710395813,
                        -0.6194435954093933,
                        -2.4776573181152344,
                        -2.3507184982299805,
                        0.3698756694793701,
                        -2.0518434047698975,
                        -1.5726983547210693,
                        -1.5274162292480469,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 60, 80)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [
                        -1.143147587776184,
                        -0.2128320187330246,
                        -0.6190732717514038,
                        -2.4692535400390625,
                        -2.3453481197357178,
                        0.36553052067756653,
                        -2.0469343662261963,
                        -1.5744705200195312,
                        -1.5216639041900635,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3

        callback_fn.has_been_called = False

        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
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
        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests/depth2img/two_cats.png"
        )
        inputs = {
            "prompt": "two tigers",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_depth2img_pndm(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_depth2img_ddim(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_lms(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_dpm(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 30
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
