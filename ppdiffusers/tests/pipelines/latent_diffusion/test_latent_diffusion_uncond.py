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

import unittest

import numpy as np
import paddle

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel
from ppdiffusers import DDIMScheduler, LDMPipeline, UNet2DModel, VQModel
from ppdiffusers.utils.testing_utils import require_paddle, slow


class LDMPipelineFastTests(unittest.TestCase):
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
            [0.827049, 1.0, 0.6244688, 0.7729403, 1.0, 0.73071766, 0.6108738, 0.9107263, 0.7249622]
        )
        tolerance = 0.01
        assert np.abs(image_slice.flatten() - expected_slice).max() < tolerance
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < tolerance


@slow
@require_paddle
class LDMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_uncond(self):
        ldm = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")
        ldm.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = ldm(generator=generator, num_inference_steps=5, output_type="numpy").images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array(
            [0.59802866, 0.61698544, 0.62753576, 0.6128236, 0.60961217, 0.617262, 0.6060791, 0.60261935, 0.6129079]
        )
        tolerance = 0.01
        assert np.abs(image_slice.flatten() - expected_slice).max() < tolerance
