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
    StableDiffusionInpaintPipeline,
    StableDiffusionInpaintPipelineLegacy,
    UNet2DConditionModel,
    UNet2DModel,
    VQModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, slow
from PIL import Image
from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer

from test_pipelines_common import PipelineTesterMixin


class StableDiffusionInpaintLegacyPipelineFastTests(PipelineTesterMixin,
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

    def test_stable_diffusion_inpaint_legacy(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.transpose([0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize(
            (32, 32))

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
            init_image=init_image,
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
            init_image=init_image,
            mask_image=mask_image,
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([
            0.0370868444442749, 0.5526567101478577, 0.3823889493942261,
            0.17967790365219116, 0.5249443054199219, 0.5843778848648071,
            0.37140119075775146, 0.615413248538971, 0.5159661173820496
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() -
                      expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_legacy_negative_prompt(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.transpose([0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize(
            (32, 32))

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
            init_image=init_image,
            mask_image=mask_image,
        )

        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([
            0.012376785278320312, 0.5630697011947632, 0.3990080952644348,
            0.18168213963508606, 0.5862641334533691, 0.6070377826690674,
            0.36853668093681335, 0.6737475991249084, 0.5175082683563232
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_legacy_num_images_per_prompt(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.transpose([0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize(
            (32, 32))

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
            init_image=init_image,
            mask_image=mask_image,
        ).images

        assert images.shape == (1, 32, 32, 3)

        # test num_images_per_prompt=1 (default) for batch of prompts
        batch_size = 2
        images = sd_pipe(
            [prompt] * batch_size,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
            mask_image=mask_image,
        ).images

        assert images.shape == (batch_size, 32, 32, 3)

        # test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            init_image=init_image,
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
            init_image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)


@slow
class StableDiffusionInpaintLegacyPipelineIntegrationTests(unittest.TestCase):

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_inpaint_legacy_pipeline(self):
        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data"
            "/overture-creations-5sI6fQgYIuo.png")
        mask_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data"
            "/overture-creations-5sI6fQgYIuo_mask.png")
        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data"
            "/red_cat_sitting_on_a_park_bench.npy")

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A red cat sitting on a park bench"

        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            mask_image=mask_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-3

    def test_stable_diffusion_inpaint_legacy_pipeline_k_lms(self):
        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data"
            "/overture-creations-5sI6fQgYIuo.png")
        mask_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data"
            "/overture-creations-5sI6fQgYIuo_mask.png")
        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data"
            "/red_cat_sitting_on_a_park_bench_k_lms.npy")

        model_id = "CompVis/stable-diffusion-v1-4"
        lms = LMSDiscreteScheduler.from_pretrained(model_id,
                                                   subfolder="scheduler")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            scheduler=lms,
            safety_checker=None,
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A red cat sitting on a park bench"

        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt,
            init_image=init_image,
            mask_image=mask_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-3

    def test_stable_diffusion_inpaint_legacy_intermediate_state(self):
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
                    -0.5472262501716614, 1.1218345165252686,
                    -0.5504080057144165, -0.9390195608139038,
                    -1.0794956684112549, 0.4064966142177582, 0.5158499479293823,
                    0.6427926421165466, -1.524450421333313
                ])
                assert np.abs(latents_slice.flatten() -
                              expected_slice).max() < 1e-3
            elif step == 37:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([
                    0.4780615568161011, 1.1574957370758057, 0.6261215806007385,
                    0.22902169823646545, 0.2551727890968323,
                    -0.14348584413528442, 0.7087294459342957,
                    -0.16012200713157654, -0.5653802156448364
                ])
                assert np.abs(latents_slice.flatten() -
                              expected_slice).max() < 1e-3

        test_callback_fn.has_been_called = False

        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data"
            "/overture-creations-5sI6fQgYIuo.png")
        mask_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data"
            "/overture-creations-5sI6fQgYIuo_mask.png")

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A red cat sitting on a park bench"

        generator = paddle.Generator().manual_seed(0)
        pipe(
            prompt=prompt,
            init_image=init_image,
            mask_image=mask_image,
            strength=0.75,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
            callback=test_callback_fn,
            callback_steps=1,
        )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 37
