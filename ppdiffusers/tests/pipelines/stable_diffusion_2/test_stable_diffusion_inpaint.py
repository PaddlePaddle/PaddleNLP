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
    PNDMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, slow


class StableDiffusionInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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

    def test_stable_diffusion_inpaint(self):
        unet = self.dummy_cond_unet_inpaint
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        text_encoder = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image_dummy = self.dummy_image.transpose([0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image_dummy)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image_dummy + 4)).convert("RGB").resize((64, 64))

        # make sure here that pndm scheduler skips prk
        sd_pipe = StableDiffusionInpaintPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=None,
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
            image=init_image,
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
            image=init_image,
            mask_image=mask_image,
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.46581971645355225,
                0.4502384066581726,
                0.2711679935455322,
                0.39340725541114807,
                0.4130115509033203,
                0.47139889001846313,
                0.35663333535194397,
                0.33137619495391846,
                0.39559948444366455,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
class StableDiffusionInpaintPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_inpaint_pipeline(self):
        init_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/init_image_sd2.png")
        mask_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/mask_sd2.png")
        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/yellow_cat_sitting_on_a_park_bench_sd2.npy"
        )

        model_id = "stabilityai/stable-diffusion-2-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        assert np.abs(expected_image - image).max() < 1e-3
