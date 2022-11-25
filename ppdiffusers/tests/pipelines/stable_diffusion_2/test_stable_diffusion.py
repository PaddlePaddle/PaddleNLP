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
import tempfile
import unittest

import numpy as np
import paddle

from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)
from ppdiffusers.utils import load_numpy, slow
from paddlenlp.transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from test_pipelines_common import PipelineTesterMixin


class StableDiffusion2PipelineFastTests(PipelineTesterMixin, unittest.TestCase):

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

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
            # SD2-specific config below
            attention_head_dim=(2, 4, 8, 8),
            use_linear_projection=True,
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
            sample_size=128,
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
            # SD2-specific config below
            text_hidden_act="gelu",
            projection_dim=512,
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

                def to(self, *args, **kwargs):
                    return self

            return Out()

        return extract

    def test_save_pretrained_from_pretrained(self):
        unet = self.dummy_cond_unet
        sample_size = unet.config.sample_size
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")
        feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt],
                         generator=generator,
                         guidance_scale=6.0,
                         num_inference_steps=2,
                         output_type="np")
        image = output.images

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd_pipe.save_pretrained(tmpdirname)
            sd_pipe = StableDiffusionPipeline.from_pretrained(tmpdirname)
        sd_pipe.set_progress_bar_config(disable=None)
        sd_pipe.unet.config.sample_size = sample_size
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt],
                         generator=generator,
                         guidance_scale=6.0,
                         num_inference_steps=2,
                         output_type="np")
        new_image = output.images

        assert np.abs(image - new_image).sum(
        ) < 1e-5, "Models don't have the same forward pass"

    def test_stable_diffusion_ddim(self):
        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
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

        generator = generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt],
                         generator=generator,
                         guidance_scale=6.0,
                         num_inference_steps=2,
                         output_type="np")
        image = output.images

        generator = generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([
            0.3399079442024231, 0.32056277990341187, 0.3508835434913635,
            0.17867285013198853, 0.24185702204704285, 0.4110303521156311,
            0.17070192098617554, 0.1501854658126831, 0.35932618379592896
        ])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() -
                      expected_slice).max() < 1e-2

    def test_stable_diffusion_pndm(self):
        unet = self.dummy_cond_unet
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
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
        generator = generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt],
                         generator=generator,
                         guidance_scale=6.0,
                         num_inference_steps=2,
                         output_type="np")

        image = output.images

        generator = generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([
            0.20569297671318054, 0.292793869972229, 0.35201555490493774,
            0.17686176300048828, 0.28506627678871155, 0.4446532726287842,
            0.17510178685188293, 0.15651720762252808, 0.35780689120292664
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() -
                      expected_slice).max() < 1e-2

    def test_stable_diffusion_k_lms(self):
        unet = self.dummy_cond_unet
        scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                         beta_end=0.012,
                                         beta_schedule="scaled_linear")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
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
        generator = generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt],
                         generator=generator,
                         guidance_scale=6.0,
                         num_inference_steps=2,
                         output_type="np")

        image = output.images

        generator = generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([
            0.3788182735443115, 0.33073586225509644, 0.35480794310569763,
            0.15712517499923706, 0.23421180248260498, 0.3990577459335327,
            0.1658017635345459, 0.1397191286087036, 0.3329782485961914
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() -
                      expected_slice).max() < 1e-2

    def test_stable_diffusion_k_euler_ancestral(self):
        unet = self.dummy_cond_unet
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
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
        generator = generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt],
                         generator=generator,
                         guidance_scale=6.0,
                         num_inference_steps=2,
                         output_type="np")

        image = output.images

        generator = generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([
            0.37895363569259644, 0.3304874002933502, 0.3544866144657135,
            0.15734165906906128, 0.23358365893363953, 0.3988724648952484,
            0.1663529872894287, 0.13962090015411377, 0.3329361081123352
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() -
                      expected_slice).max() < 1e-2

    def test_stable_diffusion_k_euler(self):
        unet = self.dummy_cond_unet
        scheduler = EulerDiscreteScheduler(beta_start=0.00085,
                                           beta_end=0.012,
                                           beta_schedule="scaled_linear")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
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
        generator = generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt],
                         generator=generator,
                         guidance_scale=6.0,
                         num_inference_steps=2,
                         output_type="np")

        image = output.images

        generator = generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([
            0.37881898880004883, 0.33073627948760986, 0.35480785369873047,
            0.15712547302246094, 0.23421168327331543, 0.3990575969219208,
            0.165802001953125, 0.13971921801567078, 0.3329782783985138
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() -
                      expected_slice).max() < 1e-2

    def test_stable_diffusion_attention_chunk(self):
        unet = self.dummy_cond_unet
        scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                         beta_end=0.012,
                                         beta_schedule="scaled_linear")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionPipeline(
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
        generator = generator = paddle.Generator().manual_seed(0)
        output_1 = sd_pipe([prompt],
                           generator=generator,
                           guidance_scale=6.0,
                           num_inference_steps=2,
                           output_type="np")

        # make sure chunking the attention yields the same result
        sd_pipe.enable_attention_slicing(slice_size=1)
        generator = generator = paddle.Generator().manual_seed(0)
        output_2 = sd_pipe([prompt],
                           generator=generator,
                           guidance_scale=6.0,
                           num_inference_steps=2,
                           output_type="np")

        assert np.abs(output_2.images.flatten() -
                      output_1.images.flatten()).max() < 1e-4


@slow
class StableDiffusion2PipelineIntegrationTests(unittest.TestCase):

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base")
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt],
                         generator=generator,
                         guidance_scale=6.0,
                         num_inference_steps=20,
                         output_type="np")

        image = output.images
        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([
            0.0788, 0.0823, 0.1091, 0.1165, 0.1263, 0.1459, 0.1317, 0.1507,
            0.1551
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_ddim(self):
        scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-base", subfolder="scheduler")
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base", scheduler=scheduler)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = generator = paddle.Generator().manual_seed(0)

        output = sd_pipe([prompt],
                         generator=generator,
                         num_inference_steps=5,
                         output_type="numpy")
        image = output.images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([
            0.0642, 0.0382, 0.0408, 0.0395, 0.0227, 0.0942, 0.0749, 0.0669,
            0.0248
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_k_lms(self):
        scheduler = LMSDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-base", subfolder="scheduler")
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base", scheduler=scheduler)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "a photograph of an astronaut riding a horse"
        generator = generator = paddle.Generator().manual_seed(0)
        image = sd_pipe([prompt],
                        generator=generator,
                        guidance_scale=7.5,
                        num_inference_steps=5,
                        output_type="numpy").images

        image_slice = image[0, 253:256, 253:256, -1]
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([
            0.0548, 0.0626, 0.0612, 0.0611, 0.0706, 0.0586, 0.0843, 0.0333,
            0.1197
        ])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_text2img_pipeline_default(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-text2img/astronaut_riding_a_horse.npy"
        )

        model_id = "stabilityai/stable-diffusion-2-base"
        pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                       safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "astronaut riding a horse"

        generator = generator = paddle.Generator().manual_seed(0)
        output = pipe(prompt=prompt,
                      strength=0.75,
                      guidance_scale=7.5,
                      generator=generator,
                      output_type="np")
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 5e-3

    def test_stable_diffusion_text2img_intermediate_state(self):
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
                    1.8606, 1.3169, -0.0691, 1.2374, -2.309, 1.077, -0.1084,
                    -0.6774, -2.9594
                ])
                assert np.abs(latents_slice.flatten() -
                              expected_slice).max() < 1e-3
            elif step == 20:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([
                    1.078, 1.1804, 1.1339, 0.4664, -0.2354, 0.6097, -0.7749,
                    -0.8784, -0.9465
                ])
                assert np.abs(latents_slice.flatten() -
                              expected_slice).max() < 1e-2

        test_callback_fn.has_been_called = False

        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base")
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "Andromeda galaxy in a bottle"

        generator = generator = paddle.Generator().manual_seed(0)
        pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            callback=test_callback_fn,
            callback_steps=1,
        )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 21
