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

from paddlenlp.transformers import CLIPTextModel
from ppdiffusers import DDIMScheduler, LDMPipeline, UNet2DModel, VQModel
from ppdiffusers.utils.testing_utils import slow


class LDMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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

    def test_inference_uncond(self):
        unet = self.dummy_uncond_unet
        scheduler = DDIMScheduler()
        vae = self.dummy_vq_model

        ldm = LDMPipeline(unet=unet, vqvae=vae, scheduler=scheduler)
        ldm.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)

        image = ldm(generator=generator, num_inference_steps=2, output_type="numpy").images

        generator = paddle.Generator().manual_seed(0)

        image_from_tuple = ldm(generator=generator, num_inference_steps=2, output_type="numpy", return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                0.8270485401153564,
                1.0,
                0.6244686841964722,
                0.772940456867218,
                1.0,
                0.7307173609733582,
                0.6108742356300354,
                0.9107264280319214,
                0.7249622344970703,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
class LDMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_uncond(self):
        ldm = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")
        ldm.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)

        image = ldm(generator=generator, num_inference_steps=5, output_type="numpy").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array(
            [
                0.5980286598205566,
                0.6169854402542114,
                0.6275357007980347,
                0.6128235459327698,
                0.6096121668815613,
                0.6172620058059692,
                0.6060792207717896,
                0.6026193499565125,
                0.6129077672958374,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
