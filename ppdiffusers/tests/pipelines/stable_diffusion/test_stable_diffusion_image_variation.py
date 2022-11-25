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

from ppdiffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImageVariationPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, slow
from paddlenlp.transformers import CLIPVisionModelWithProjection

from test_pipelines_common import PipelineTesterMixin


class StableDiffusionImageVariationPipelineFastTests(PipelineTesterMixin,
                                                     unittest.TestCase):

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

        image = floats_tensor((batch_size, num_channels) + sizes,
                              rng=random.Random(0))
        return image

    @property
    def dummy_cond_unet(self):
        paddle.seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=64,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
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
    def dummy_image_encoder(self):
        paddle.seed(0)
        config = dict(
            vision_embed_dim=32,
            projection_dim=32,
            vision_mlp_ratio=1,
            vision_heads=4,
            vision_layers=5,
            image_resolution=32,
            vision_patch_size=4,
        )
        model = CLIPVisionModelWithProjection(**config)
        model.eval()
        return model

    @property
    def dummy_extractor(self):

        def extract(*args, **kwargs):

            class Out:

                def __init__(self):
                    self.pixel_values = paddle.ones([0])

                def to(self, *args, **kwargs):
                    return self

            return Out()

        return extract

    def test_stable_diffusion_img_variation_default_case(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        image_encoder = self.dummy_image_encoder

        init_image = self.dummy_image

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImageVariationPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            image_encoder=image_encoder,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe(
            init_image,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
        )

        image = output.images

        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            init_image,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]

        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([
            0.9133340120315552, 1.0, 0.4939959645271301, 0.4945739209651947,
            0.3666399419307709, 0.2726040184497833, 0.4451484680175781,
            0.4383566975593567, 0.5677292943000793
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
        assert np.abs(image_from_tuple_slice.flatten() -
                      expected_slice).max() < 1e-3

    def test_stable_diffusion_img_variation_multiple_images(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        image_encoder = self.dummy_image_encoder

        init_image = self.dummy_image.tile([2, 1, 1, 1])

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImageVariationPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            image_encoder=image_encoder,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe(
            init_image,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
        )

        image = output.images

        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 128, 128, 3)
        expected_slice = np.array([
            0.46606171131134033, 0.36306852102279663, 0.5599685907363892,
            0.5636163353919983, 0.30924463272094727, 0.26244693994522095,
            0.4429951608181, 0.47115442156791687, 0.41289669275283813
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img_variation_num_images_per_prompt(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        image_encoder = self.dummy_image_encoder

        init_image = self.dummy_image

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImageVariationPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            image_encoder=image_encoder,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        # test num_images_per_prompt=1 (default)
        images = sd_pipe(
            init_image,
            num_inference_steps=2,
            output_type="np",
        ).images

        assert images.shape == (1, 128, 128, 3)

        # test num_images_per_prompt=1 (default) for batch of images
        batch_size = 2
        images = sd_pipe(
            init_image.tile([batch_size, 1, 1, 1]),
            num_inference_steps=2,
            output_type="np",
        ).images

        assert images.shape == (batch_size, 128, 128, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            init_image,
            num_inference_steps=2,
            output_type="np",
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (num_images_per_prompt, 128, 128, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        images = sd_pipe(
            init_image.tile([batch_size, 1, 1, 1]),
            num_inference_steps=2,
            output_type="np",
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (batch_size * num_images_per_prompt, 128, 128, 3)


@slow
class StableDiffusionImageVariationPipelineIntegrationTests(unittest.TestCase):

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_img_variation_pipeline_default(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/vermeer.jpg"
        )
        init_image = init_image.resize((512, 512))
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/vermeer.npy"
        )

        model_id = "fusing/sd-image-variations-diffusers"
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            model_id,
            safety_checker=None,
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            init_image,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        # img2img is flaky across GPUs even in fp32, so using MAE here
        assert np.abs(expected_image - image).max() < 1e-3

    def test_stable_diffusion_img_variation_intermediate_state(self):
        number_of_steps = 0

        def test_callback_fn(step: int, timestep: int,
                             latents: paddle.Tensor) -> None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([
                    1.83, 1.293, -0.09705, 1.256, -2.293, 1.091, -0.0809, -0.65,
                    -2.953
                ])
                assert np.abs(latents_slice.flatten() -
                              expected_slice).max() < 1e-3
            elif step == 37:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([
                    2.285, 2.703, 1.969, 0.696, -1.323, 0.9253, -0.5464, -1.521,
                    -2.537
                ])
                assert np.abs(latents_slice.flatten() -
                              expected_slice).max() < 1e-2

        test_callback_fn.has_been_called = False

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg")
        init_image = init_image.resize((512, 512))

        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "fusing/sd-image-variations-diffusers", )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        generator = paddle.Generator().manual_seed(0)
        pipe(
            init_image,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
            callback=test_callback_fn,
            callback_steps=1,
        )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 51
