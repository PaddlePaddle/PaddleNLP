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

import unittest

import numpy as np
import paddle
from test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LDMTextToImagePipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils.testing_utils import slow


class LDMTextToImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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

    def test_inference_text2img(self):
        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler()
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        ldm = LDMTextToImagePipeline(vqvae=vae, bert=bert, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
        ldm.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        generator = paddle.Generator().manual_seed(0)

        image = ldm(
            [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="numpy"
        ).images

        generator = paddle.Generator().manual_seed(0)

        image_from_tuple = ldm(
            [prompt],
            generator=generator,
            guidance_scale=6.0,
            num_inference_steps=2,
            output_type="numpy",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.20027577877044678,
                0.0171242356300354,
                0.3928355872631073,
                0.1479446291923523,
                0.12596771121025085,
                0.45699092745780945,
                0.31846195459365845,
                0.1955386996269226,
                0.4729262590408325,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
class LDMTextToImagePipelineIntegrationTests(unittest.TestCase):
    def test_inference_text2img(self):
        ldm = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        ldm.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)

        image = ldm(
            [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=20, output_type="numpy"
        ).images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array(
            [
                0.14945372939109802,
                0.13170909881591797,
                0.13690760731697083,
                0.14863789081573486,
                0.140784353017807,
                0.1294558048248291,
                0.14045551419258118,
                0.12387925386428833,
                0.125020831823349,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_text2img_fast(self):
        ldm = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        ldm.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)

        image = ldm(prompt, generator=generator, num_inference_steps=1, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array(
            [
                0.0,
                0.21720609068870544,
                0.1733270287513733,
                0.09142807126045227,
                0.2129485011100769,
                0.25848573446273804,
                0.28228187561035156,
                0.22558268904685974,
                0.3113105893135071,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
