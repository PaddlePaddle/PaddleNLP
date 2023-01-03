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
from test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    CycleDiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    UNet2DModel,
    VQModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, slow


class CycleDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
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

    def test_stable_diffusion_cycle(self):
        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
        )
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        # make sure here that pndm scheduler skips prk
        sd_pipe = CycleDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        source_prompt = "An astronaut riding a horse"
        prompt = "An astronaut riding an elephant"
        init_image = self.dummy_image

        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe(
            prompt=prompt,
            source_prompt=source_prompt,
            generator=generator,
            num_inference_steps=2,
            image=init_image,
            eta=0.1,
            strength=0.8,
            guidance_scale=3,
            source_guidance_scale=1,
            output_type="np",
        )
        images = output.images

        image_slice = images[0, -3:, -3:, -1]

        assert images.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [
                0.054250508546829224,
                0.7774901390075684,
                0.7094265222549438,
                0.1572849154472351,
                0.978983461856842,
                0.49742749333381653,
                0.362673282623291,
                0.6486804485321045,
                0.45295289158821106,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


# TODO junnyu
@slow
class CycleDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_cycle_diffusion_pipeline(self):
        init_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/black_colored_car.png")
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/cycle-diffusion/blue_colored_car.npy"
        )
        init_image = init_image.resize((512, 512))

        model_id = "CompVis/stable-diffusion-v1-4"
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = CycleDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, safety_checker=None)

        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        source_prompt = "A black colored car"
        prompt = "A blue colored car"

        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt,
            source_prompt=source_prompt,
            image=init_image,
            num_inference_steps=100,
            eta=0.1,
            strength=0.85,
            guidance_scale=3,
            source_guidance_scale=1,
            generator=generator,
            output_type="np",
        )
        image = output.images

        assert np.abs(image - expected_image).max() < 1e-2
