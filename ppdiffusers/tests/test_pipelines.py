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
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock

import numpy as np
import paddle
import PIL
import safetensors.torch
from parameterized import parameterized
from PIL import Image
from requests.exceptions import HTTPError

from paddlenlp.transformers import (
    CLIPFeatureExtractor,
    CLIPModel,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
)
from ppdiffusers import (
    AutoencoderKL,
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    DiffusionPipeline,
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
from ppdiffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from ppdiffusers.utils import CONFIG_NAME, WEIGHTS_NAME, floats_tensor, nightly, slow
from ppdiffusers.utils.testing_utils import (
    CaptureLogger,
    get_tests_dir,
    require_paddle_gpu,
)


class DownloadTests(unittest.TestCase):
    def test_download_only_pytorch(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = DiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None, cache_dir=tmpdirname
            )
            all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname, os.listdir(tmpdirname)[0], "snapshots"))]
            files = [item for sublist in all_root_files for item in sublist]
            assert not any(f.endswith(".msgpack") for f in files)
            assert not any(f.endswith(".safetensors") for f in files)

    def test_returned_cached_folder(self):
        prompt = "hello"
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        _, local_path = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None, return_cached_folder=True
        )
        pipe_2 = StableDiffusionPipeline.from_pretrained(local_path)
        generator = paddle.Generator().manual_seed(0)
        out = pipe(prompt, num_inference_steps=2, generator=generator, output_type="numpy").images
        generator = paddle.Generator().manual_seed(0)
        out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator, output_type="numpy").images
        assert np.max(np.abs(out - out_2)) < 0.001

    def test_download_safetensors(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = DiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe-safetensors", safety_checker=None, cache_dir=tmpdirname
            )
            all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname, os.listdir(tmpdirname)[0], "snapshots"))]
            files = [item for sublist in all_root_files for item in sublist]
            assert not any(f.endswith(".bin") for f in files)

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
        assert np.max(np.abs(out - out_2)) < 0.001

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
            generator = paddle.Generator().manual_seed(0)
            out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator, output_type="numpy").images
        assert np.max(np.abs(out - out_2)) < 0.001

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
        assert np.max(np.abs(out - out_2)) < 0.001

    def test_cached_files_are_used_when_no_internet(self):
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}
        orig_pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        orig_comps = {k: v for k, v in orig_pipe.components.items() if hasattr(v, "parameters")}
        with mock.patch("requests.request", return_value=response_mock):
            pipe = StableDiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None, local_files_only=True
            )
            comps = {k: v for k, v in pipe.components.items() if hasattr(v, "parameters")}
        for m1, m2 in zip(orig_comps.values(), comps.values()):
            for p1, p2 in zip(m1.parameters(), m2.parameters()):
                if p1.data.ne(p2.data).sum() > 0:
                    assert False, "Parameters not the same!"

    def test_download_from_variant_folder(self):
        for safe_avail in [False, True]:
            import ppdiffusers

            ppdiffusers.utils.import_utils._safetensors_available = safe_avail
            other_format = ".bin" if safe_avail else ".safetensors"
            with tempfile.TemporaryDirectory() as tmpdirname:
                StableDiffusionPipeline.from_pretrained(
                    "hf-internal-testing/stable-diffusion-all-variants", cache_dir=tmpdirname
                )
                all_root_files = [
                    t[-1] for t in os.walk(os.path.join(tmpdirname, os.listdir(tmpdirname)[0], "snapshots"))
                ]
                files = [item for sublist in all_root_files for item in sublist]
                assert len(files) == 15, f"We should only download 15 files, not {len(files)}"
                assert not any(f.endswith(other_format) for f in files)
                assert not any(len(f.split(".")) == 3 for f in files)
        ppdiffusers.utils.import_utils._safetensors_available = True

    def test_download_variant_all(self):
        for safe_avail in [False, True]:
            import ppdiffusers

            ppdiffusers.utils.import_utils._safetensors_available = safe_avail
            other_format = ".bin" if safe_avail else ".safetensors"
            this_format = ".safetensors" if safe_avail else ".bin"
            variant = "fp16"
            with tempfile.TemporaryDirectory() as tmpdirname:
                StableDiffusionPipeline.from_pretrained(
                    "hf-internal-testing/stable-diffusion-all-variants", cache_dir=tmpdirname, variant=variant
                )
                all_root_files = [
                    t[-1] for t in os.walk(os.path.join(tmpdirname, os.listdir(tmpdirname)[0], "snapshots"))
                ]
                files = [item for sublist in all_root_files for item in sublist]
                assert len(files) == 15, f"We should only download 15 files, not {len(files)}"
                assert len([f for f in files if f.endswith(f"{variant}{this_format}")]) == 4
                assert not any(f.endswith(this_format) and not f.endswith(f"{variant}{this_format}") for f in files)
                assert not any(f.endswith(other_format) for f in files)
        ppdiffusers.utils.import_utils._safetensors_available = True

    def test_download_variant_partly(self):
        for safe_avail in [False, True]:
            import ppdiffusers

            ppdiffusers.utils.import_utils._safetensors_available = safe_avail
            other_format = ".bin" if safe_avail else ".safetensors"
            this_format = ".safetensors" if safe_avail else ".bin"
            variant = "no_ema"
            with tempfile.TemporaryDirectory() as tmpdirname:
                StableDiffusionPipeline.from_pretrained(
                    "hf-internal-testing/stable-diffusion-all-variants", cache_dir=tmpdirname, variant=variant
                )
                snapshots = os.path.join(tmpdirname, os.listdir(tmpdirname)[0], "snapshots")
                all_root_files = [t[-1] for t in os.walk(snapshots)]
                files = [item for sublist in all_root_files for item in sublist]
                unet_files = os.listdir(os.path.join(snapshots, os.listdir(snapshots)[0], "unet"))
                assert len(files) == 15, f"We should only download 15 files, not {len(files)}"
                assert f"diffusion_pytorch_model.{variant}{this_format}" in unet_files
                assert len([f for f in files if f.endswith(f"{variant}{this_format}")]) == 1
                assert sum(f.endswith(this_format) and not f.endswith(f"{variant}{this_format}") for f in files) == 3
                assert not any(f.endswith(other_format) for f in files)
        ppdiffusers.utils.import_utils._safetensors_available = True

    def test_download_broken_variant(self):
        for safe_avail in [False, True]:
            import ppdiffusers

            ppdiffusers.utils.import_utils._safetensors_available = safe_avail
            for variant in [None, "no_ema"]:
                with self.assertRaises(OSError) as error_context:
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        StableDiffusionPipeline.from_pretrained(
                            "hf-internal-testing/stable-diffusion-broken-variants",
                            cache_dir=tmpdirname,
                            variant=variant,
                        )
                assert "Error no file name" in str(error_context.exception)
            with tempfile.TemporaryDirectory() as tmpdirname:
                pipe = StableDiffusionPipeline.from_pretrained(
                    "hf-internal-testing/stable-diffusion-broken-variants", cache_dir=tmpdirname, variant="fp16"
                )
                assert pipe is not None
                snapshots = os.path.join(tmpdirname, os.listdir(tmpdirname)[0], "snapshots")
                all_root_files = [t[-1] for t in os.walk(snapshots)]
                files = [item for sublist in all_root_files for item in sublist]
                assert len(files) == 15, f"We should only download 15 files, not {len(files)}"
        ppdiffusers.utils.import_utils._safetensors_available = True


class CustomPipelineTests(unittest.TestCase):
    def test_load_custom_pipeline(self):
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/ppdiffusers-dummy-pipeline"
        )
        pipeline = pipeline
        assert pipeline.__class__.__name__ == "CustomPipeline"

    def test_load_custom_github(self):
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline="one_step_unet", custom_revision="main"
        )
        with paddle.no_grad():
            output = pipeline()
        assert output.size() == output.sum()
        del sys.modules["diffusers_modules.git.one_step_unet"]
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline="one_step_unet", custom_revision="0.10.2"
        )
        with paddle.no_grad():
            output = pipeline()
        assert output.size() != output.sum()
        assert pipeline.__class__.__name__ == "UnetSchedulerOneForwardPipeline"

    def test_run_custom_pipeline(self):
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/ppdiffusers-dummy-pipeline"
        )
        pipeline = pipeline
        images, output_str = pipeline(num_inference_steps=2, output_type="np")
        assert images[0].shape == (1, 32, 32, 3)
        assert output_str == "This is a test"

    def test_local_custom_pipeline_repo(self):
        local_custom_pipeline_path = get_tests_dir("fixtures/custom_pipeline")
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline=local_custom_pipeline_path
        )
        pipeline = pipeline
        images, output_str = pipeline(num_inference_steps=2, output_type="np")
        assert pipeline.__class__.__name__ == "CustomLocalPipeline"
        assert images[0].shape == (1, 32, 32, 3)
        assert output_str == "This is a local test"

    def test_local_custom_pipeline_file(self):
        local_custom_pipeline_path = get_tests_dir("fixtures/custom_pipeline")
        local_custom_pipeline_path = os.path.join(local_custom_pipeline_path, "what_ever.py")
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline=local_custom_pipeline_path
        )
        pipeline = pipeline
        images, output_str = pipeline(num_inference_steps=2, output_type="np")
        assert pipeline.__class__.__name__ == "CustomLocalPipeline"
        assert images[0].shape == (1, 32, 32, 3)
        assert output_str == "This is a local test"

    @slow
    @require_paddle_gpu
    def test_load_pipeline_from_git(self):
        clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id)
        clip_model = CLIPModel.from_pretrained(clip_model_id, paddle_dtype=paddle.float16)
        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            custom_pipeline="clip_guided_stable_diffusion",
            clip_model=clip_model,
            feature_extractor=feature_extractor,
            paddle_dtype=paddle.float16,
        )
        pipeline.enable_attention_slicing()
        pipeline = pipeline
        assert pipeline.__class__.__name__ == "CLIPGuidedStableDiffusion"
        image = pipeline("a prompt", num_inference_steps=2, output_type="np").images[0]
        assert image.shape == (512, 512, 3)


class PipelineFastTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()
        import ppdiffusers

        ppdiffusers.utils.import_utils._safetensors_available = True

    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = 32, 32
        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
        return image

    def dummy_uncond_unet(self, sample_size=32):
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

    def dummy_cond_unet(self, sample_size=32):
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
        config = CLIPTextConfig(
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
        return CLIPTextModel(config).eval()

    @property
    def dummy_extractor(self):
        def extract(*args, **kwargs):
            class Out:
                def __init__(self):
                    self.pixel_values = paddle.ones(shape=[0])

                def to(self, device):
                    self.pixel_values
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
        generator = paddle.Generator().manual_seed(0)
        out_image = pipeline(generator=generator, num_inference_steps=2, output_type="np").images
        sample_size = (sample_size, sample_size) if isinstance(sample_size, int) else sample_size
        assert out_image.shape == (1, *sample_size, 3)

    def test_stable_diffusion_components(self):
        """Test that components property works correctly"""
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        image = self.dummy_image().cpu().transpose(perm=[0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((32, 32))
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
            image=init_image,
            mask_image=mask_image,
        ).images
        image_img2img = img2img(
            [prompt], generator=generator, num_inference_steps=2, output_type="np", image=init_image
        ).images
        image_text2img = text2img([prompt], generator=generator, num_inference_steps=2, output_type="np").images
        assert image_inpaint.shape == (1, 32, 32, 3)
        assert image_img2img.shape == (1, 32, 32, 3)
        assert image_text2img.shape == (1, 64, 64, 3)

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
            feature_extractor=self.dummy_extractor,
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
            feature_extractor=self.dummy_extractor,
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
            feature_extractor=self.dummy_extractor,
        )
        ddim_config = sd.scheduler.config
        sd.scheduler = LMSDiscreteScheduler.from_config(ddim_config)
        sd.scheduler = DDIMScheduler.from_config(sd.scheduler.config)
        ddim_config_2 = sd.scheduler.config
        ddim_config_2 = {k: v for k, v in ddim_config_2.items() if k in ddim_config}
        assert dict(ddim_config) == dict(ddim_config_2)

    def test_save_safe_serialization(self):
        pipeline = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipeline.save_pretrained(tmpdirname, safe_serialization=True)
            vae_path = os.path.join(tmpdirname, "vae", "diffusion_pytorch_model.safetensors")
            assert os.path.exists(vae_path), f"Could not find {vae_path}"
            _ = safetensors.torch.load_file(vae_path)
            unet_path = os.path.join(tmpdirname, "unet", "diffusion_pytorch_model.safetensors")
            assert os.path.exists(unet_path), f"Could not find {unet_path}"
            _ = safetensors.torch.load_file(unet_path)
            text_encoder_path = os.path.join(tmpdirname, "text_encoder", "model.safetensors")
            assert os.path.exists(text_encoder_path), f"Could not find {text_encoder_path}"
            _ = safetensors.torch.load_file(text_encoder_path)
            pipeline = StableDiffusionPipeline.from_pretrained(tmpdirname)
            assert pipeline.unet is not None
            assert pipeline.vae is not None
            assert pipeline.text_encoder is not None
            assert pipeline.scheduler is not None
            assert pipeline.feature_extractor is not None

    def test_no_pytorch_download_when_doing_safetensors(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = StableDiffusionPipeline.from_pretrained(
                "hf-internal-testing/ppdiffusers-stable-diffusion-tiny-all", cache_dir=tmpdirname
            )
            path = os.path.join(
                tmpdirname,
                "models--hf-internal-testing--ppdiffusers-stable-diffusion-tiny-all",
                "snapshots",
                "07838d72e12f9bcec1375b0482b80c1d399be843",
                "unet",
            )
            assert os.path.exists(os.path.join(path, "diffusion_pytorch_model.safetensors"))
            assert not os.path.exists(os.path.join(path, "diffusion_pytorch_model.bin"))

    def test_no_safetensors_download_when_doing_pytorch(self):
        import ppdiffusers

        ppdiffusers.utils.import_utils._safetensors_available = False
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = StableDiffusionPipeline.from_pretrained(
                "hf-internal-testing/ppdiffusers-stable-diffusion-tiny-all", cache_dir=tmpdirname
            )
            path = os.path.join(
                tmpdirname,
                "models--hf-internal-testing--ppdiffusers-stable-diffusion-tiny-all",
                "snapshots",
                "07838d72e12f9bcec1375b0482b80c1d399be843",
                "unet",
            )
            assert not os.path.exists(os.path.join(path, "diffusion_pytorch_model.safetensors"))
            assert os.path.exists(os.path.join(path, "diffusion_pytorch_model.bin"))
        ppdiffusers.utils.import_utils._safetensors_available = True

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
            sd = StableDiffusionPipeline.from_pretrained(
                tmpdirname, feature_extractor=None, safety_checker=None, requires_safety_checker=False
            )
            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)
        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname)
            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)
            orig_sd.save_pretrained(tmpdirname)
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
        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, feature_extractor=self.dummy_extractor)
            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor != (None, None)
            sd = StableDiffusionPipeline.from_pretrained(
                tmpdirname,
                feature_extractor=self.dummy_extractor,
                safety_checker=unet,
                requires_safety_checker=[True, True],
            )
            assert sd.config.requires_safety_checker == [True, True]
            assert sd.config.safety_checker != (None, None)
            assert sd.config.feature_extractor != (None, None)
        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, feature_extractor=self.dummy_extractor)
            assert sd.config.requires_safety_checker == [True, True]
            assert sd.config.safety_checker != (None, None)
            assert sd.config.feature_extractor != (None, None)


@slow
@require_paddle_gpu
class PipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_smart_download(self):
        model_id = "hf-internal-testing/unet-pipeline-dummy"
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = DiffusionPipeline.from_pretrained(model_id, cache_dir=tmpdirname, force_download=True)
            local_repo_name = "--".join(["models"] + model_id.split("/"))
            snapshot_dir = os.path.join(tmpdirname, local_repo_name, "snapshots")
            snapshot_dir = os.path.join(snapshot_dir, os.listdir(snapshot_dir)[0])
            assert os.path.isfile(os.path.join(snapshot_dir, DiffusionPipeline.config_name))
            assert os.path.isfile(os.path.join(snapshot_dir, CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, SCHEDULER_CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, WEIGHTS_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "scheduler", SCHEDULER_CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "unet", WEIGHTS_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "unet", WEIGHTS_NAME))
            assert not os.path.isfile(os.path.join(snapshot_dir, "big_array.npy"))

    def test_warning_unused_kwargs(self):
        model_id = "hf-internal-testing/unet-pipeline-dummy"
        logger = logging.get_logger("ppdiffusers.pipelines")
        with tempfile.TemporaryDirectory() as tmpdirname:
            with CaptureLogger(logger) as cap_logger:
                DiffusionPipeline.from_pretrained(model_id, not_used=True, cache_dir=tmpdirname, force_download=True)
        assert (
            cap_logger.out.strip().split("\n")[-1]
            == "Keyword arguments {'not_used': True} are not expected by DDPMPipeline and will be ignored."
        )

    def test_from_save_pretrained(self):
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
        image = ddpm(generator=generator, num_inference_steps=5, output_type="numpy").images
        generator = paddle.Generator().manual_seed(0)
        new_image = new_ddpm(generator=generator, num_inference_steps=5, output_type="numpy").images
        assert np.abs(image - new_image).sum() < 1e-05, "Models don't give the same forward pass"

    def test_from_pretrained_hub(self):
        model_path = "google/ddpm-cifar10-32"
        scheduler = DDPMScheduler(num_train_timesteps=10)
        ddpm = DDPMPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm.set_progress_bar_config(disable=None)
        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub = ddpm_from_hub
        ddpm_from_hub.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=5, output_type="numpy").images
        generator = paddle.Generator().manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, num_inference_steps=5, output_type="numpy").images
        assert np.abs(image - new_image).sum() < 1e-05, "Models don't give the same forward pass"

    def test_from_pretrained_hub_pass_model(self):
        model_path = "google/ddpm-cifar10-32"
        scheduler = DDPMScheduler(num_train_timesteps=10)
        unet = UNet2DModel.from_pretrained(model_path)
        ddpm_from_hub_custom_model = DiffusionPipeline.from_pretrained(model_path, unet=unet, scheduler=scheduler)
        ddpm_from_hub_custom_model = ddpm_from_hub_custom_model
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)
        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = ddpm_from_hub_custom_model(generator=generator, num_inference_steps=5, output_type="numpy").images
        generator = paddle.Generator().manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, num_inference_steps=5, output_type="numpy").images
        assert np.abs(image - new_image).sum() < 1e-05, "Models don't give the same forward pass"

    def test_output_format(self):
        model_path = "google/ddpm-cifar10-32"
        scheduler = DDIMScheduler.from_pretrained(model_path)
        pipe = DDIMPipeline.from_pretrained(model_path, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=None)
        images = pipe(output_type="numpy").images
        assert images.shape == (1, 32, 32, 3)
        assert isinstance(images, np.ndarray)
        images = pipe(output_type="pil", num_inference_steps=4).images
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], PIL.Image.Image)
        images = pipe(num_inference_steps=4).images
        assert isinstance(images, list)
        assert isinstance(images[0], PIL.Image.Image)


@nightly
@require_paddle_gpu
class PipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

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
            use_clipped_model_output=True,
        ).images
        assert np.abs(ddpm_images - ddim_images).max() < 0.1
