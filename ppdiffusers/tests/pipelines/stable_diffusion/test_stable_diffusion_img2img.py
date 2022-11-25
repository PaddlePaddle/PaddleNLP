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
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
    UNet2DModel,
    VQModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, slow
from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer

from test_pipelines_common import PipelineTesterMixin


class StableDiffusionImg2ImgPipelineFastTests(PipelineTesterMixin,
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
        model = CLIPTextModel(**config)
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

    def test_stable_diffusion_img2img_default_case(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
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
            init_image=init_image,
        )

        image = output.images

        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([
            0.5996609926223755, 0.3400457799434662, 0.4609818756580353,
            0.21939468383789062, 0.18144932389259338, 0.36151182651519775,
            0.5850329995155334, 0.462869793176651, 0.5071303844451904
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
        assert np.abs(image_from_tuple_slice.flatten() -
                      expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_negative_prompt(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
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
            init_image=init_image,
        )
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([
            0.7830642461776733, 0.39025136828422546, 0.5853854417800903,
            0.35690179467201233, 0.21060478687286377, 0.44461506605148315,
            0.6436530351638794, 0.5138731598854065, 0.6071125268936157
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_multiple_init_images(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image.tile([2, 1, 1, 1])

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = 2 * ["A painting of a squirrel eating a burger"]
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe(
            prompt,
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
        )

        image = output.images

        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 32, 32, 3)
        expected_slice = np.array([
            0.7116804718971252, 0.46600550413131714, 0.6155763268470764,
            0.8523228168487549, 0.47249341011047363, 0.393263578414917,
            0.904474675655365, 0.5542261600494385, 0.5442584156990051
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_k_lms(self):
        unet = self.dummy_cond_unet
        scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                         beta_end=0.012,
                                         beta_schedule="scaled_linear")

        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
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
            init_image=init_image,
        )
        image = output.images

        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            return_dict=False,
        )
        image_from_tuple = output[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([
            0.10615921020507812, 0.7351866364479065, 0.6672630310058594,
            0.15450340509414673, 0.8547155261039734, 0.4804829955101013,
            0.39658865332603455, 0.6143652200698853, 0.4508455991744995
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
        assert np.abs(image_from_tuple_slice.flatten() -
                      expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_num_images_per_prompt(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        init_image = self.dummy_image

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionImg2ImgPipeline(
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
            init_image=init_image,
        ).images

        assert images.shape == (1, 32, 32, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
        ).images

        assert images.shape == (batch_size, 32, 32, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (num_images_per_prompt, 32, 32, 3)

        # test num_images_per_prompt for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)


@slow
class StableDiffusionImg2ImgPipelineIntegrationTests(unittest.TestCase):

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_img2img_pipeline_default(self):
        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))
        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/fantasy_landscape.npy"
        )

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            safety_checker=None,
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 768, 3)
        # img2img is flaky across GPUs even in fp32, so using MAE here
        assert np.abs(expected_image - image).max() < 1e-3

    def test_stable_diffusion_img2img_pipeline_k_lms(self):
        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))
        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/fantasy_landscape_k_lms.npy"
        )

        model_id = "CompVis/stable-diffusion-v1-4"
        lms = LMSDiscreteScheduler.from_pretrained(model_id,
                                                   subfolder="scheduler")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            scheduler=lms,
            safety_checker=None,
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 768, 3)
        assert np.abs(expected_image - image).max() < 1e-3

    def test_stable_diffusion_img2img_pipeline_ddim(self):
        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))
        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/fantasy_landscape_ddim.npy"
        )

        model_id = "CompVis/stable-diffusion-v1-4"
        ddim = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            scheduler=ddim,
            safety_checker=None,
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 768, 3)
        assert np.abs(expected_image - image).max() < 1e-3

    def test_stable_diffusion_img2img_intermediate_state(self):
        number_of_steps = 0

        def test_callback_fn(step: int, timestep: int,
                             latents: paddle.Tensor) -> None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([
                    0.9075717329978943, -0.017079180106520653,
                    0.48078930377960205, 0.2827966809272766, 0.5796141624450684,
                    1.4848742485046387, 0.5332279205322266, 1.9802331924438477,
                    0.05084720626473427
                ])
                assert np.abs(latents_slice.flatten() -
                              expected_slice).max() < 1e-3
            elif step == 37:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([
                    0.7074431777000427, 0.784945011138916, 0.8420439958572388,
                    1.8224966526031494, 1.792901873588562, 1.9434930086135864,
                    1.3401929140090942, 1.656121850013733, 1.2673498392105103
                ])
                assert np.abs(latents_slice.flatten() -
                              expected_slice).max() < 1e-2

        test_callback_fn.has_been_called = False

        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = paddle.Generator().manual_seed(0)
        pipe(
            prompt=prompt,
            init_image=init_image,
            strength=0.75,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
            callback=test_callback_fn,
            callback_steps=1,
        )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 38
