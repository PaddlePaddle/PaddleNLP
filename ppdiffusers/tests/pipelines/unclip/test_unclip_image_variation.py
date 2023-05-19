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
import random
import unittest

import numpy as np
import paddle
from ppdiffusers_test.pipeline_params import (
    IMAGE_VARIATION_BATCH_PARAMS,
    IMAGE_VARIATION_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import (
    PipelineTesterMixin,
    assert_mean_pixel_difference,
)

from paddlenlp.transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)
from ppdiffusers import (
    DiffusionPipeline,
    UnCLIPImageVariationPipeline,
    UnCLIPScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)
from ppdiffusers.pipelines.unclip.text_proj import UnCLIPTextProjModel
from ppdiffusers.utils import floats_tensor, slow
from ppdiffusers.utils.testing_utils import load_image, require_paddle_gpu


class UnCLIPImageVariationPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = UnCLIPImageVariationPipeline
    params = IMAGE_VARIATION_PARAMS - {"height", "width", "guidance_scale"}
    batch_params = IMAGE_VARIATION_BATCH_PARAMS
    required_optional_params = frozenset(
        ["generator", "return_dict", "decoder_num_inference_steps", "super_res_num_inference_steps"]
    )

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
    def dummy_image_encoder(self):
        paddle.seed(0)
        config = CLIPVisionConfig(
            hidden_size=self.text_embedder_hidden_size,
            projection_dim=self.text_embedder_hidden_size,
            num_hidden_layers=5,
            num_attention_heads=4,
            image_size=32,
            intermediate_size=37,
            patch_size=1,
        )
        return CLIPVisionModelWithProjection(config)

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
        decoder = self.dummy_decoder
        text_proj = self.dummy_text_proj
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        super_res_first = self.dummy_super_res_first
        super_res_last = self.dummy_super_res_last
        decoder_scheduler = UnCLIPScheduler(
            variance_type="learned_range", prediction_type="epsilon", num_train_timesteps=1000
        )
        super_res_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log", prediction_type="epsilon", num_train_timesteps=1000
        )
        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
        image_encoder = self.dummy_image_encoder
        return {
            "decoder": decoder,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_proj": text_proj,
            "feature_extractor": feature_extractor,
            "image_encoder": image_encoder,
            "super_res_first": super_res_first,
            "super_res_last": super_res_last,
            "decoder_scheduler": decoder_scheduler,
            "super_res_scheduler": super_res_scheduler,
        }

    def test_xformers_attention_forwardGenerator_pass(self):
        pass

    def get_dummy_inputs(self, seed=0, pil_image=True):
        input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        generator = paddle.Generator().manual_seed(seed)

        if pil_image:
            input_image = input_image * 0.5 + 0.5
            input_image = input_image.clip(min=0, max=1)
            input_image = input_image.cpu().transpose(perm=[0, 2, 3, 1]).float().numpy()
            input_image = DiffusionPipeline.numpy_to_pil(input_image)[0]
        return {
            "image": input_image,
            "generator": generator,
            "decoder_num_inference_steps": 2,
            "super_res_num_inference_steps": 2,
            "output_type": "np",
        }

    def test_unclip_image_variation_input_tensor(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipeline_inputs = self.get_dummy_inputs(pil_image=False)
        output = pipe(**pipeline_inputs)
        image = output.images
        tuple_pipeline_inputs = self.get_dummy_inputs(pil_image=False)
        image_from_tuple = pipe(**tuple_pipeline_inputs, return_dict=False)[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                2.7585030e-03,
                2.6383996e-04,
                9.9801058e-01,
                2.6383996e-04,
                9.9531418e-01,
                9.9220645e-01,
                3.6702752e-03,
                9.9970925e-01,
                9.9973619e-01,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01

    def test_unclip_image_variation_input_image(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipeline_inputs = self.get_dummy_inputs(pil_image=True)
        output = pipe(**pipeline_inputs)
        image = output.images
        tuple_pipeline_inputs = self.get_dummy_inputs(pil_image=True)
        image_from_tuple = pipe(**tuple_pipeline_inputs, return_dict=False)[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [
                5.2168965e-04,
                9.9861604e-01,
                9.9755847e-01,
                9.9804187e-01,
                9.9411416e-01,
                9.9248302e-01,
                9.9973619e-01,
                9.9777901e-01,
                9.9973619e-01,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01

    def test_unclip_image_variation_input_list_images(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipeline_inputs = self.get_dummy_inputs(pil_image=True)
        pipeline_inputs["image"] = [pipeline_inputs["image"], pipeline_inputs["image"]]
        output = pipe(**pipeline_inputs)
        image = output.images
        tuple_pipeline_inputs = self.get_dummy_inputs(pil_image=True)
        tuple_pipeline_inputs["image"] = [tuple_pipeline_inputs["image"], tuple_pipeline_inputs["image"]]
        image_from_tuple = pipe(**tuple_pipeline_inputs, return_dict=False)[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (2, 64, 64, 3)
        expected_slice = np.array(
            [
                5.2201748e-04,
                9.9861759e-01,
                9.9755961e-01,
                9.9804127e-01,
                9.9411547e-01,
                9.9248385e-01,
                9.9973619e-01,
                9.9777836e-01,
                9.9973619e-01,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01

    def test_unclip_image_variation_input_num_images_per_prompt(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipeline_inputs = self.get_dummy_inputs(pil_image=True)
        pipeline_inputs["image"] = [pipeline_inputs["image"], pipeline_inputs["image"]]
        output = pipe(**pipeline_inputs, num_images_per_prompt=2)
        image = output.images
        tuple_pipeline_inputs = self.get_dummy_inputs(pil_image=True)
        tuple_pipeline_inputs["image"] = [tuple_pipeline_inputs["image"], tuple_pipeline_inputs["image"]]
        image_from_tuple = pipe(**tuple_pipeline_inputs, num_images_per_prompt=2, return_dict=False)[0]
        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
        assert image.shape == (4, 64, 64, 3)
        expected_slice = np.array(
            [
                5.2204728e-04,
                9.9861759e-01,
                9.9755961e-01,
                9.9804127e-01,
                9.9411547e-01,
                9.9248385e-01,
                9.9973619e-01,
                9.9777836e-01,
                9.9973619e-01,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01

    def test_unclip_passed_image_embed(self):
        class DummyScheduler:
            init_noise_sigma = 1

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        dtype = pipe.decoder.dtype
        batch_size = 1
        shape = (batch_size, pipe.decoder.in_channels, pipe.decoder.sample_size, pipe.decoder.sample_size)
        decoder_latents = pipe.prepare_latents(
            shape, dtype=dtype, generator=generator, latents=None, scheduler=DummyScheduler()
        )
        shape = (
            batch_size,
            pipe.super_res_first.in_channels // 2,
            pipe.super_res_first.sample_size,
            pipe.super_res_first.sample_size,
        )
        super_res_latents = pipe.prepare_latents(
            shape, dtype=dtype, generator=generator, latents=None, scheduler=DummyScheduler()
        )
        pipeline_inputs = self.get_dummy_inputs(pil_image=False)
        img_out_1 = pipe(
            **pipeline_inputs, decoder_latents=decoder_latents, super_res_latents=super_res_latents
        ).images
        pipeline_inputs = self.get_dummy_inputs(pil_image=False)
        image = pipeline_inputs.pop("image")
        image_embeddings = pipe.image_encoder(image).image_embeds
        img_out_2 = pipe(
            **pipeline_inputs,
            decoder_latents=decoder_latents,
            super_res_latents=super_res_latents,
            image_embeddings=image_embeddings,
        ).images
        assert np.abs(img_out_1 - img_out_2).max() < 0.0001

    def test_attention_slicing_forward_pass(self):
        test_max_difference = False
        self._test_attention_slicing_forward_pass(test_max_difference=test_max_difference)

    def test_inference_batch_single_identical(self):
        test_max_difference = False
        relax_max_difference = True
        additional_params_copy_to_batched_inputs = [
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
            "decoder_num_inference_steps",
            "super_res_num_inference_steps",
        ]

        self._test_inference_batch_consistent(
            additional_params_copy_to_batched_inputs=additional_params_copy_to_batched_inputs
        )


@slow
@require_paddle_gpu
class UnCLIPImageVariationPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_unclip_image_variation_karlo(self):
        input_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unclip/cat.png"
        )
        expected_image = np.array([[0.09096909, 0.13343304, 0.26244187], [0.15095001, 0.19459972, 0.3182609]])
        # TODO(wugaosheng): test this function
        pipeline = UnCLIPImageVariationPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations")
        pipeline.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        output = pipeline(input_image, generator=generator, output_type="np")
        image = output.images[0]
        assert image.shape == (256, 256, 3)

        assert_mean_pixel_difference(image[0][0:2], expected_image)
