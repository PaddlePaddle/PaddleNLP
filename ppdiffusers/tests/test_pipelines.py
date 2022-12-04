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
import json
import os
import random
import shutil
import tempfile
import unittest

import numpy as np
import paddle
import PIL
from parameterized import parameterized
from PIL import Image

from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UNet2DModel,
    logging,
)
from ppdiffusers.pipeline_utils import DiffusionPipeline
from ppdiffusers.utils import floats_tensor, slow
from ppdiffusers.utils.testing_utils import CaptureLogger


def test_progress_bar(capsys):
    model = UNet2DModel(
        block_out_channels=(32, 64),
        layers_per_block=2,
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    )
    scheduler = DDPMScheduler(num_train_timesteps=10)

    ddpm = DDPMPipeline(model, scheduler)
    ddpm(output_type="numpy").images
    captured = capsys.readouterr()
    assert "10/10" in captured.err, "Progress bar has to be displayed"

    ddpm.set_progress_bar_config(disable=True)
    ddpm(output_type="numpy").images
    captured = capsys.readouterr()
    assert captured.err == "", "Progress bar should be disabled"


class DownloadTests(unittest.TestCase):
    def test_download_no_safety_checker(self):
        prompt = "hello"
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        generator = paddle.Generator().manual_seed(0)
        out = pipe(prompt, num_inference_steps=2, generator=generator, output_type="numpy").images

        pipe_2 = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
        generator = paddle.Generator().manual_seed(0)
        out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator, output_type="numpy").images

        assert np.max(np.abs(out - out_2)) < 1e-3

    def test_load_no_safety_checker_explicit_locally(self):
        prompt = "hello"
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        generator = paddle.Generator().manual_seed(0)
        out = pipe(prompt, num_inference_steps=2, generator=generator, output_type="numpy").images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe_2 = StableDiffusionPipeline.from_pretrained(tmpdirname, safety_checker=None)
            pipe_2 = pipe_2

            generator = paddle.Generator().manual_seed(0)

            out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator, output_type="numpy").images

        assert np.max(np.abs(out - out_2)) < 1e-3

    def test_load_no_safety_checker_default_locally(self):
        prompt = "hello"
        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
        generator = paddle.Generator().manual_seed(0)
        out = pipe(prompt, num_inference_steps=2, generator=generator, output_type="numpy").images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe_2 = StableDiffusionPipeline.from_pretrained(tmpdirname)

            generator = paddle.Generator().manual_seed(0)

            out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator, output_type="numpy").images

        assert np.max(np.abs(out - out_2)) < 1e-3


class PipelineFastTests(unittest.TestCase):
    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
        return image

    def dummy_uncond_unet(self, sample_size=64):
        paddle.seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=sample_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

    def dummy_cond_unet(self, sample_size=64):
        paddle.seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=sample_size,
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

                def to(self, *args, **kwargs):
                    return self

            return Out()

        return extract

    @parameterized.expand(
        [
            [DDIMScheduler, DDIMPipeline, 32],
            [DDPMScheduler, DDPMPipeline, 32],
            [DDIMScheduler, DDIMPipeline, (32, 64)],
            [DDPMScheduler, DDPMPipeline, (64, 32)],
        ]
    )
    def test_uncond_unet_components(self, scheduler_fn=DDPMScheduler, pipeline_fn=DDPMPipeline, sample_size=32):
        unet = self.dummy_uncond_unet(sample_size)
        scheduler = scheduler_fn()
        pipeline = pipeline_fn(unet, scheduler)

        # Device type MPS is not supported for torch.Generator() api.
        generator = paddle.Generator().manual_seed(0)

        out_image = pipeline(
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images
        sample_size = (sample_size, sample_size) if isinstance(sample_size, int) else sample_size
        assert out_image.shape == (1, *sample_size, 3)

    def test_components(self):
        """Test that components property works correctly"""
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image_dummy = self.dummy_image().cpu().transpose([0, 2, 3, 1])[0]
        image = Image.fromarray(np.uint8(image_dummy)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image_dummy + 4)).convert("RGB").resize((32, 32))

        # make sure here that pndm scheduler skips prk
        inpaint = StableDiffusionInpaintPipelineLegacy(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        img2img = StableDiffusionImg2ImgPipeline(**inpaint.components)
        text2img = StableDiffusionPipeline(**inpaint.components)

        prompt = "A painting of a squirrel eating a burger"

        generator = paddle.Generator().manual_seed(0)

        image_inpaint = inpaint(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            image=image,
            mask_image=mask_image,
        ).images
        image_img2img = img2img(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            image=image,
        ).images
        image_text2img = text2img(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images

        assert image_inpaint.shape == (1, 32, 32, 3)
        assert image_img2img.shape == (1, 32, 32, 3)
        assert image_text2img.shape == (1, 128, 128, 3)

    def test_set_scheduler(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor(),
        )

        sd.scheduler = DDIMScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, DDIMScheduler)
        sd.scheduler = DDPMScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, DDPMScheduler)
        sd.scheduler = PNDMScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, PNDMScheduler)
        sd.scheduler = LMSDiscreteScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, LMSDiscreteScheduler)
        sd.scheduler = EulerDiscreteScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, EulerDiscreteScheduler)
        sd.scheduler = EulerAncestralDiscreteScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, EulerAncestralDiscreteScheduler)
        sd.scheduler = DPMSolverMultistepScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, DPMSolverMultistepScheduler)

    def test_set_scheduler_consistency(self):
        unet = self.dummy_cond_unet()
        pndm = PNDMScheduler.from_config("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler")
        ddim = DDIMScheduler.from_config("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=pndm,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor(),
        )

        pndm_config = sd.scheduler.config
        sd.scheduler = DDPMScheduler.from_config(pndm_config)
        sd.scheduler = PNDMScheduler.from_config(sd.scheduler.config)
        pndm_config_2 = sd.scheduler.config
        pndm_config_2 = {k: v for k, v in pndm_config_2.items() if k in pndm_config}

        assert dict(pndm_config) == dict(pndm_config_2)

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=ddim,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor(),
        )

        ddim_config = sd.scheduler.config
        sd.scheduler = LMSDiscreteScheduler.from_config(ddim_config)
        sd.scheduler = DDIMScheduler.from_config(sd.scheduler.config)
        ddim_config_2 = sd.scheduler.config
        ddim_config_2 = {k: v for k, v in ddim_config_2.items() if k in ddim_config}

        assert dict(ddim_config) == dict(ddim_config_2)

    def test_optional_components(self):
        unet = self.dummy_cond_unet()
        pndm = PNDMScheduler.from_config("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        orig_sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=pndm,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=unet,
            feature_extractor=self.dummy_extractor,
        )
        sd = orig_sd

        assert sd.config.requires_safety_checker is True

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)

            # Test that passing None works
            sd = StableDiffusionPipeline.from_pretrained(
                tmpdirname, feature_extractor=None, safety_checker=None, requires_safety_checker=False
            )

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

            sd.save_pretrained(tmpdirname)

            # Test that loading previous None works
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

            orig_sd.save_pretrained(tmpdirname)

            # Test that loading without any directory works
            shutil.rmtree(os.path.join(tmpdirname, "safety_checker"))
            with open(os.path.join(tmpdirname, sd.config_name)) as f:
                config = json.load(f)
                config["safety_checker"] = [None, None]
            with open(os.path.join(tmpdirname, sd.config_name), "w") as f:
                json.dump(config, f)

            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, requires_safety_checker=False)
            sd.save_pretrained(tmpdirname)
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

            # Test that loading from deleted model index works
            with open(os.path.join(tmpdirname, sd.config_name)) as f:
                config = json.load(f)
                del config["safety_checker"]
                del config["feature_extractor"]
            with open(os.path.join(tmpdirname, sd.config_name), "w") as f:
                json.dump(config, f)

            sd = StableDiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

            sd.save_pretrained(tmpdirname)

            # Test that partially loading works
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, feature_extractor=self.dummy_extractor)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor != (None, None)

            # Test that partially loading works
            sd = StableDiffusionPipeline.from_pretrained(
                tmpdirname,
                feature_extractor=self.dummy_extractor,
                safety_checker=unet,
                requires_safety_checker=[True, True],
            )

            assert sd.config.requires_safety_checker == [True, True]
            assert sd.config.safety_checker != (None, None)
            assert sd.config.feature_extractor != (None, None)

            sd.save_pretrained(tmpdirname)
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, feature_extractor=self.dummy_extractor)

            assert sd.config.requires_safety_checker == [True, True]
            assert sd.config.safety_checker != (None, None)
            assert sd.config.feature_extractor != (None, None)


@slow
class PipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_warning_unused_kwargs(self):
        model_id = "hf-internal-testing/unet-pipeline-dummy"
        logger = logging.get_logger("ppdiffusers.pipeline_utils")
        with tempfile.TemporaryDirectory() as tmpdirname:
            with CaptureLogger(logger) as cap_logger:
                DiffusionPipeline.from_pretrained(
                    model_id,
                    not_used=True,
                    cache_dir=tmpdirname,
                )

        assert (
            cap_logger.out
            == "Keyword arguments {'not_used': True} are not expected by DDPMPipeline and will be ignored.\n"
        )

    def test_from_save_pretrained(self):
        # 1. Load models
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        schedular = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline(model, schedular)
        ddpm.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPMPipeline.from_pretrained(tmpdirname)

        generator = paddle.Generator().manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        generator = paddle.Generator().manual_seed(0)
        new_image = new_ddpm(generator=generator, output_type="numpy").images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    def test_from_pretrained_hub(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm.set_progress_bar_config(disable=None)

        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images

        generator = paddle.Generator().manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, output_type="numpy").images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    def test_from_pretrained_hub_pass_model(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        # pass unet into DiffusionPipeline
        unet = UNet2DModel.from_pretrained(model_path)
        ddpm_from_hub_custom_model = DiffusionPipeline.from_pretrained(model_path, unet=unet, scheduler=scheduler)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)

        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        image = ddpm_from_hub_custom_model(generator=generator, output_type="numpy").images

        generator = paddle.Generator().manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, output_type="numpy").images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't give the same forward pass"

    def test_output_format(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDIMScheduler.from_pretrained(model_path)
        pipe = DDIMPipeline.from_pretrained(model_path, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        images = pipe(generator=generator, output_type="numpy").images
        assert images.shape == (1, 32, 32, 3)
        assert isinstance(images, np.ndarray)

        images = pipe(generator=generator, output_type="pil", num_inference_steps=4).images
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], PIL.Image.Image)

        # use PIL by default
        images = pipe(generator=generator, num_inference_steps=4).images
        assert isinstance(images, list)
        assert isinstance(images[0], PIL.Image.Image)

    def test_ddpm_ddim_equality_batched(self):
        seed = 0
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        ddpm_scheduler = DDPMScheduler()
        ddim_scheduler = DDIMScheduler()

        ddpm = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler)
        ddpm.set_progress_bar_config(disable=None)

        ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
        ddim.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(seed)
        ddpm_images = ddpm(batch_size=2, generator=generator, output_type="numpy").images

        generator = paddle.Generator().manual_seed(seed)
        ddim_images = ddim(
            batch_size=2,
            generator=generator,
            num_inference_steps=1000,
            eta=1.0,
            output_type="numpy",
            use_clipped_model_output=True,  # Need this to make DDIM match DDPM
        ).images

        # the values aren't exactly equal, but the images look the same visually
        assert np.abs(ddpm_images - ddim_images).max() < 1e-1
