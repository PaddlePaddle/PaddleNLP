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
import unittest

import numpy as np
import paddle
from ppdiffusers_test.pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import (
    PipelineTesterMixin,
    assert_mean_pixel_difference,
)

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
from ppdiffusers.utils import slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class UnCLIPPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = UnCLIPPipeline
    params = TEXT_TO_IMAGE_PARAMS - {
        "negative_prompt",
        "height",
        "width",
        "negative_prompt_embeds",
        "guidance_scale",
        "prompt_embeds",
        "cross_attention_kwargs",
    }
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    required_optional_params = frozenset(
        [
            "generator",
            "return_dict",
            "prior_num_inference_steps",
            "decoder_num_inference_steps",
            "super_res_num_inference_steps",
        ]
    )
    test_xformers_attention = False

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
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=self.text_embedder_hidden_size,
            projection_dim=self.text_embedder_hidden_size,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModelWithProjection(config)

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
            "sample_size": 32,
            "in_channels": 3,
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
            "sample_size": 64,
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
        paddle.seed(seed=1)
        model = UNet2DModel(**self.dummy_super_res_kwargs)
        return model

    def get_dummy_components(self):
        prior = self.dummy_prior
        decoder = self.dummy_decoder
        text_proj = self.dummy_text_proj
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        super_res_first = self.dummy_super_res_first
        super_res_last = self.dummy_super_res_last
        prior_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log", prediction_type="sample", num_train_timesteps=1000, clip_sample_range=5.0
        )
        decoder_scheduler = UnCLIPScheduler(
            variance_type="learned_range", prediction_type="epsilon", num_train_timesteps=1000
        )
        super_res_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log", prediction_type="epsilon", num_train_timesteps=1000
        )
        components = {
            "prior": prior,
            "decoder": decoder,
            "text_proj": text_proj,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "super_res_first": super_res_first,
            "super_res_last": super_res_last,
            "prior_scheduler": prior_scheduler,
            "decoder_scheduler": decoder_scheduler,
            "super_res_scheduler": super_res_scheduler,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": "horse",
            "generator": generator,
            "prior_num_inference_steps": 2,
            "decoder_num_inference_steps": 2,
            "super_res_num_inference_steps": 2,
            "output_type": "numpy",
        }
        return inputs

    def test_unclip(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        output = pipe(**self.get_dummy_inputs())
        image = output.images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                2.6383996e-04,
                9.9658674e-01,
                1.1275411e-03,
                2.6383996e-04,
                2.6383996e-04,
                9.9702907e-01,
                9.9973619e-01,
                9.9545717e-01,
                2.6383996e-04,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01

    def test_unclip_passed_text_embed(self):
        class DummyScheduler:
            init_noise_sigma = 1

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        prior = components["prior"]
        decoder = components["decoder"]
        super_res_first = components["super_res_first"]
        tokenizer = components["tokenizer"]
        text_encoder = components["text_encoder"]
        generator = paddle.Generator().manual_seed(0)
        dtype = prior.dtype
        batch_size = 1
        shape = batch_size, prior.config.embedding_dim
        prior_latents = pipe.prepare_latents(
            shape, dtype=dtype, generator=generator, latents=None, scheduler=DummyScheduler()
        )
        shape = (batch_size, decoder.in_channels, decoder.sample_size, decoder.sample_size)
        decoder_latents = pipe.prepare_latents(
            shape, dtype=dtype, generator=generator, latents=None, scheduler=DummyScheduler()
        )
        shape = (
            batch_size,
            super_res_first.in_channels // 2,
            super_res_first.sample_size,
            super_res_first.sample_size,
        )
        super_res_latents = pipe.prepare_latents(
            shape, dtype=dtype, generator=generator, latents=None, scheduler=DummyScheduler()
        )
        pipe.set_progress_bar_config(disable=None)
        prompt = "this is a prompt example"
        generator = paddle.Generator().manual_seed(0)
        output = pipe(
            [prompt],
            generator=generator,
            prior_num_inference_steps=2,
            decoder_num_inference_steps=2,
            super_res_num_inference_steps=2,
            prior_latents=prior_latents,
            decoder_latents=decoder_latents,
            super_res_latents=super_res_latents,
            output_type="np",
        )
        image = output.images
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_attention_mask=True,
            return_tensors="pd",
        )
        text_model_output = text_encoder(text_inputs.input_ids)
        text_attention_mask = text_inputs.attention_mask
        generator = paddle.Generator().manual_seed(0)
        image_from_text = pipe(
            generator=generator,
            prior_num_inference_steps=2,
            decoder_num_inference_steps=2,
            super_res_num_inference_steps=2,
            prior_latents=prior_latents,
            decoder_latents=decoder_latents,
            super_res_latents=super_res_latents,
            text_model_output=text_model_output,
            text_attention_mask=text_attention_mask,
            output_type="np",
        )[0]
        assert np.abs(image - image_from_text).max() < 0.0001

    def test_attention_slicing_forward_pass(self):
        test_max_difference = False
        self._test_attention_slicing_forward_pass(test_max_difference=test_max_difference)

    def test_inference_batch_single_identical(self):
        test_max_difference = False
        relax_max_difference = True
        additional_params_copy_to_batched_inputs = [
            "prior_num_inference_steps",
            "decoder_num_inference_steps",
            "super_res_num_inference_steps",
        ]

        self._test_inference_batch_single_identical(
            test_max_difference=test_max_difference,
            relax_max_difference=relax_max_difference,
            additional_params_copy_to_batched_inputs=additional_params_copy_to_batched_inputs,
        )

    def test_inference_batch_consistent(self):
        additional_params_copy_to_batched_inputs = [
            "prior_num_inference_steps",
            "decoder_num_inference_steps",
            "super_res_num_inference_steps",
        ]

        self._test_inference_batch_consistent(
            additional_params_copy_to_batched_inputs=additional_params_copy_to_batched_inputs
        )


@slow
@require_paddle_gpu
class UnCLIPPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_unclip_karlo(self):
        # Hard code image
        expected_image = np.array([[0.73281264, 0.69175875, 0.64672112], [0.71919304, 0.65395129, 0.60436499]])
        pipeline = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha")
        pipeline.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        output = pipeline("horse", generator=generator, output_type="np")
        image = output.images[0]
        assert image.shape == (256, 256, 3)
        assert_mean_pixel_difference(image[0][0:2], expected_image)
