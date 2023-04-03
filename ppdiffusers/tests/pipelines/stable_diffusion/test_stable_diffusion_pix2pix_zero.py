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
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    StableDiffusionPix2PixZeroPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import load_image, slow
from ppdiffusers.utils.testing_utils import load_pt, require_paddle_gpu


def to_paddle(x):
    if hasattr(x, "numpy"):
        x = x.numpy()
    return paddle.to_tensor(x)


# we use SGD optimizer in this pipeline, so the result is not stable!
class StableDiffusionPix2PixZeroPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionPix2PixZeroPipeline

    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    @classmethod
    def setUpClass(cls):
        cls.source_embeds = to_paddle(
            load_pt(
                "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/src_emb_0.pt"
            )
        )

        cls.target_embeds = to_paddle(
            load_pt(
                "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/tgt_emb_0.pt"
            )
        )

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
        scheduler = DDIMScheduler()
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
            "inverse_scheduler": None,
            "caption_generator": None,
            "caption_processor": None,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "cross_attention_guidance_amount": 0.15,
            "source_embeds": self.source_embeds,
            "target_embeds": self.target_embeds,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_pix2pix_zero_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.17479098, 0.374185, 0.47584057, 0.30907357, 0.17638746, 0.42003536, 0.1912728, 0.2048004, 0.40212077]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05

    def test_stable_diffusion_pix2pix_zero_negative_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.18526673, 0.37720186, 0.43527505, 0.40688735, 0.24448359, 0.4591321, 0.25540462, 0.22007707, 0.39219254]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05

    def test_stable_diffusion_pix2pix_zero_euler(self):
        components = self.get_dummy_components()
        components["scheduler"] = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.30776912, 0.21245027, 0.28754437, 0.33886075, 0.30828524, 0.3670439, 0.35691, 0.22877613, 0.33611426]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05

    def test_stable_diffusion_pix2pix_zero_ddpm(self):
        components = self.get_dummy_components()
        components["scheduler"] = DDPMScheduler()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.17479098, 0.374185, 0.47584057, 0.30907357, 0.17638746, 0.42003536, 0.1912728, 0.2048004, 0.40212077]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05

    def test_stable_diffusion_pix2pix_zero_num_images_per_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPix2PixZeroPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs).images
        assert images.shape == (1, 64, 64, 3)
        num_images_per_prompt = 2
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images
        assert images.shape == (num_images_per_prompt, 64, 64, 3)
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * batch_size
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images
        assert images.shape == (batch_size * num_images_per_prompt, 64, 64, 3)

    # Non-determinism caused by the scheduler optimizing the latent inputs during inference
    @unittest.skip("non-deterministic pipeline")
    def test_inference_batch_single_identical(self):
        return super().test_inference_batch_single_identical()


@slow
@require_paddle_gpu
class StableDiffusionPix2PixZeroPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @classmethod
    def setUpClass(cls):
        cls.source_embeds = to_paddle(
            load_pt("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat.pt")
        )

        cls.target_embeds = to_paddle(
            load_pt("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/dog.pt")
        )

    def get_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed=seed)
        inputs = {
            "prompt": "turn him into a cyborg",
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "cross_attention_guidance_amount": 0.15,
            "source_embeds": self.source_embeds,
            "target_embeds": self.target_embeds,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_pix2pix_zero_default(self):
        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, paddle_dtype=paddle.float16
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.8129883, 0.81933594, 0.80371094, 0.8105469, 0.8076172, 0.80566406, 0.81884766, 0.8330078, 0.82470703]
        )
        assert np.abs(expected_slice - image_slice).max() < 0.05

    def test_stable_diffusion_pix2pix_zero_k_lms(self):
        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, paddle_dtype=paddle.float16
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05053711])
        assert np.abs(expected_slice - image_slice).max() < 0.05

    def test_stable_diffusion_pix2pix_zero_intermediate_state(self):
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
                        0.93444633,
                        1.1613252,
                        0.7700033,
                        0.18847837,
                        -1.17147,
                        0.07546477,
                        0.06142269,
                        -0.8030814,
                        -0.59692276,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [
                        0.93180454,
                        1.1606954,
                        0.7721853,
                        0.18454231,
                        -1.1679069,
                        0.07357024,
                        0.06213593,
                        -0.80399096,
                        -0.5937987,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05

        callback_fn.has_been_called = False
        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, paddle_dtype=paddle.float16
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 3


@slow
@require_paddle_gpu
class InversionPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @classmethod
    def setUpClass(cls):
        raw_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png"
        )

        raw_image = raw_image.convert("RGB").resize((512, 512))

        cls.raw_image = raw_image

    def test_stable_diffusion_pix2pix_inversion(self):
        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, paddle_dtype=paddle.float16
        )
        pipe.inverse_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        caption = "a photography of a cat with flowers"
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        output = pipe.invert(caption, image=self.raw_image, generator=generator, num_inference_steps=10)
        inv_latents = output[0]
        image_slice = inv_latents[0, -3:, -3:, -1].flatten()
        assert tuple(inv_latents.shape) == (1, 4, 64, 64)
        expected_slice = np.array([0.8877, 0.0587, 0.77, -1.6035, -0.5962, 0.4827, -0.6265, 1.0498, -0.8599])
        assert np.abs(expected_slice - image_slice.cpu().numpy()).max() < 0.05

    def test_stable_diffusion_pix2pix_full(self):
        pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, paddle_dtype=paddle.float16
        )
        pipe.inverse_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        caption = "a photography of a cat with flowers"
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        output = pipe.invert(caption, image=self.raw_image, generator=generator)
        inv_latents = output[0]
        source_prompts = 4 * ["a cat sitting on the street", "a cat playing in the field", "a face of a cat"]
        target_prompts = 4 * ["a dog sitting on the street", "a dog playing in the field", "a face of a dog"]
        source_embeds = pipe.get_embeds(source_prompts)
        target_embeds = pipe.get_embeds(target_prompts)
        image = pipe(
            caption,
            source_embeds=source_embeds,
            target_embeds=target_embeds,
            num_inference_steps=50,
            cross_attention_guidance_amount=0.15,
            generator=generator,
            latents=inv_latents,
            negative_prompt=caption,
            output_type="np",
        ).images

        image_slice = image[0, -3:, -3:, -1].flatten()
        expected_slice = np.array(
            [
                0.64208984375,
                0.65673828125,
                0.650390625,
                0.6513671875,
                0.646484375,
                0.6650390625,
                0.6513671875,
                0.6640625,
                0.66796875,
            ]
        )
        max_diff = np.abs(image_slice - expected_slice).max()
        assert max_diff < 0.05
