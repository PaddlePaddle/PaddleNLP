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
import unittest

import numpy as np
import paddle

from paddlenlp.transformers import (
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from ppdiffusers import (
    PriorTransformer,
    UnCLIPPipeline,
    UnCLIPScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)
from ppdiffusers.pipelines.unclip.text_proj import UnCLIPTextProjModel
from ppdiffusers.utils import load_numpy, slow


class UnCLIPPipelineFastTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @property
    def text_embedder_hidden_size(self):
        return 32

    @property
    def time_input_dim(self):
        return 32

    @property
    def block_out_channels_0(self):
        return self.time_input_dim

    @property
    def time_embed_dim(self):
        return self.time_input_dim * 4

    @property
    def cross_attention_dim(self):
        return 100

    @property
    def dummy_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        return tokenizer

    @property
    def dummy_text_encoder(self):
        paddle.seed(0)
        config = dict(
            text_embed_dim=self.text_embedder_hidden_size,
            projection_dim=self.text_embedder_hidden_size,
            text_heads=4,
            text_layers=5,
            vocab_size=1000,
        )
        config = CLIPTextConfig.from_dict(config)
        model = CLIPTextModelWithProjection(config)
        model.eval()
        return model

    @property
    def dummy_prior(self):
        paddle.seed(0)

        model_kwargs = {
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "embedding_dim": self.text_embedder_hidden_size,
            "num_layers": 1,
        }

        model = PriorTransformer(**model_kwargs)
        return model

    @property
    def dummy_text_proj(self):
        paddle.seed(0)

        model_kwargs = {
            "clip_embeddings_dim": self.text_embedder_hidden_size,
            "time_embed_dim": self.time_embed_dim,
            "cross_attention_dim": self.cross_attention_dim,
        }

        model = UnCLIPTextProjModel(**model_kwargs)
        return model

    @property
    def dummy_decoder(self):
        paddle.seed(0)

        model_kwargs = {
            "sample_size": 64,
            # RGB in channels
            "in_channels": 3,
            # Out channels is double in channels because predicts mean and variance
            "out_channels": 6,
            "down_block_types": ("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
            "up_block_types": ("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
            "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
            "block_out_channels": (self.block_out_channels_0, self.block_out_channels_0 * 2),
            "layers_per_block": 1,
            "cross_attention_dim": self.cross_attention_dim,
            "attention_head_dim": 4,
            "resnet_time_scale_shift": "scale_shift",
            "class_embed_type": "identity",
        }

        model = UNet2DConditionModel(**model_kwargs)
        return model

    @property
    def dummy_super_res_kwargs(self):
        return {
            "sample_size": 128,
            "layers_per_block": 1,
            "down_block_types": ("ResnetDownsampleBlock2D", "ResnetDownsampleBlock2D"),
            "up_block_types": ("ResnetUpsampleBlock2D", "ResnetUpsampleBlock2D"),
            "block_out_channels": (self.block_out_channels_0, self.block_out_channels_0 * 2),
            "in_channels": 6,
            "out_channels": 3,
        }

    @property
    def dummy_super_res_first(self):
        paddle.seed(0)

        model = UNet2DModel(**self.dummy_super_res_kwargs)
        return model

    @property
    def dummy_super_res_last(self):
        # seeded differently to get different unet than `self.dummy_super_res_first`
        paddle.seed(1)

        model = UNet2DModel(**self.dummy_super_res_kwargs)
        return model

    def test_unclip(self):

        prior = self.dummy_prior
        decoder = self.dummy_decoder
        text_proj = self.dummy_text_proj
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        super_res_first = self.dummy_super_res_first
        super_res_last = self.dummy_super_res_last

        prior_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="sample",
            num_train_timesteps=1000,
            clip_sample_range=5.0,
        )

        decoder_scheduler = UnCLIPScheduler(
            variance_type="learned_range",
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )

        super_res_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )

        pipe = UnCLIPPipeline(
            prior=prior,
            decoder=decoder,
            text_proj=text_proj,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            super_res_first=super_res_first,
            super_res_last=super_res_last,
            prior_scheduler=prior_scheduler,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )

        pipe.set_progress_bar_config(disable=None)

        prompt = "horse"

        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            [prompt],
            generator=generator,
            prior_num_inference_steps=2,
            decoder_num_inference_steps=2,
            super_res_num_inference_steps=2,
            output_type="np",
        )
        image = output.images

        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = pipe(
            [prompt],
            generator=generator,
            prior_num_inference_steps=2,
            decoder_num_inference_steps=2,
            super_res_num_inference_steps=2,
            output_type="np",
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)

        expected_slice = np.array(
            [
                0.9997361898422241,
                0.0002638399600982666,
                0.9997361898422241,
                0.9997361898422241,
                0.9997361898422241,
                0.9898782968521118,
                0.0002638399600982666,
                0.9997361898422241,
                0.9997361898422241,
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
class UnCLIPPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_unclip_karlo(self):
        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests/unclip/karlo_v1_alpha_horse.npy"
        )

        pipeline = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha")
        pipeline.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        output = pipeline(
            "horse",
            num_images_per_prompt=1,
            generator=generator,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (256, 256, 3)
        assert np.abs(expected_image - image).max() < 1e-2
