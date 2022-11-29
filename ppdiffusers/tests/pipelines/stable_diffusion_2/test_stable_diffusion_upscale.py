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
from PIL import Image
from test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionUpscalePipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, slow


class StableDiffusionUpscalePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
    def dummy_cond_unet_upscale(self):
        paddle.seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=7,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            # SD2-specific config below
            attention_head_dim=8,
            use_linear_projection=True,
            only_cross_attention=(True, True, False),
            num_class_embeds=100,
        )
        return model

    @property
    def dummy_vae(self):
        paddle.seed(0)
        model = AutoencoderKL(
            block_out_channels=[32, 32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
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
            # SD2-specific config below
            text_hidden_act="gelu",
            projection_dim=512,
        )
        model = CLIPTextModel(**config)
        model.eval()
        return model

    def test_stable_diffusion_upscale(self):
        unet = self.dummy_cond_unet_upscale
        low_res_scheduler = DDPMScheduler()
        scheduler = DDIMScheduler(prediction_type="v_prediction")
        vae = self.dummy_vae
        text_encoder = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image.transpose([0, 2, 3, 1])[0]
        low_res_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionUpscalePipeline(
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            max_noise_level=350,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe(
            [prompt],
            image=low_res_image,
            generator=generator,
            guidance_scale=6.0,
            noise_level=20,
            num_inference_steps=2,
            output_type="np",
        )

        image = output.images

        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe(
            [prompt],
            image=low_res_image,
            generator=generator,
            guidance_scale=6.0,
            noise_level=20,
            num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        expected_height_width = low_res_image.size[0] * 4
        assert image.shape == (1, expected_height_width, expected_height_width, 3)
        expected_slice = np.array(
            [
                0.9216755628585815,
                0.7778909206390381,
                0.7246097326278687,
                0.6616445183753967,
                0.5916030406951904,
                0.5970571041107178,
                0.5546892881393433,
                0.5396133065223694,
                0.6180068254470825,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
class StableDiffusionUpscalePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_upscale_pipeline(self):
        image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/low_res_cat.png")
        expected_image = load_numpy("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/upsampled_cat.npy")

        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "a cat sitting on a park bench"

        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=image,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-3
