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

import paddle
from ppdiffusers_test.pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    PriorTransformer,
    StableUnCLIPPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import (
    StableUnCLIPImageNormalizer,
)


class StableUnCLIPPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableUnCLIPPipeline
    test_xformers_attention = False
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS

    def get_dummy_components(self):
        embedder_hidden_size = 32
        embedder_projection_dim = embedder_hidden_size
        paddle.seed(0)
        prior_tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        paddle.seed(0)
        prior_text_encoder = CLIPTextModelWithProjection(
            CLIPTextConfig(
                bos_token_id=0,
                eos_token_id=2,
                hidden_size=embedder_hidden_size,
                projection_dim=embedder_projection_dim,
                intermediate_size=37,
                layer_norm_eps=1e-05,
                num_attention_heads=4,
                num_hidden_layers=5,
                pad_token_id=1,
                vocab_size=1000,
            )
        )
        paddle.seed(0)
        prior = PriorTransformer(
            num_attention_heads=2, attention_head_dim=12, embedding_dim=embedder_projection_dim, num_layers=1
        )
        paddle.seed(0)
        prior_scheduler = DDPMScheduler(
            variance_type="fixed_small_log",
            prediction_type="sample",
            num_train_timesteps=1000,
            clip_sample=True,
            clip_sample_range=5.0,
            beta_schedule="squaredcos_cap_v2",
        )
        paddle.seed(0)
        image_normalizer = StableUnCLIPImageNormalizer(embedding_dim=embedder_hidden_size)
        image_noising_scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2")
        paddle.seed(0)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        paddle.seed(0)
        text_encoder = CLIPTextModel(
            CLIPTextConfig(
                bos_token_id=0,
                eos_token_id=2,
                hidden_size=embedder_hidden_size,
                projection_dim=32,
                intermediate_size=37,
                layer_norm_eps=1e-05,
                num_attention_heads=4,
                num_hidden_layers=5,
                pad_token_id=1,
                vocab_size=1000,
            )
        )
        paddle.seed(0)
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(32, 64),
            attention_head_dim=(2, 4),
            class_embed_type="projection",
            projection_class_embeddings_input_dim=embedder_projection_dim * 2,
            cross_attention_dim=embedder_hidden_size,
            layers_per_block=1,
            upcast_attention=True,
            use_linear_projection=True,
        )
        paddle.seed(0)
        scheduler = DDIMScheduler(
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="v_prediction",
            set_alpha_to_one=False,
            steps_offset=1,
        )
        paddle.seed(0)
        vae = AutoencoderKL()
        components = {
            "prior_tokenizer": prior_tokenizer,
            "prior_text_encoder": prior_text_encoder,
            "prior": prior,
            "prior_scheduler": prior_scheduler,
            "image_normalizer": image_normalizer,
            "image_noising_scheduler": image_noising_scheduler,
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "prior_num_inference_steps": 2,
            "output_type": "numpy",
        }
        return inputs

    def test_attention_slicing_forward_pass(self):
        test_max_difference = False
        self._test_attention_slicing_forward_pass(test_max_difference=test_max_difference)

    def test_inference_batch_single_identical(self):
        test_max_difference = False
        self._test_inference_batch_single_identical(test_max_difference=test_max_difference)


# @slow
# @require_paddle_gpu
# class StableUnCLIPPipelineIntegrationTests(unittest.TestCase):

#     def tearDown(self):
#         super().tearDown()
#         gc.collect()
#         paddle.device.cuda.empty_cache()

#     def test_stable_unclip(self):
#         expected_image = load_numpy(
#             'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/stable_unclip_2_1_l_anime_turtle_fp16.npy'
#             )
#         pipe = StableUnCLIPPipeline.from_pretrained(
#             'fusing/stable-unclip-2-1-l')
#         pipe.set_progress_bar_config(disable=None)
#         generator = paddle.Generator().manual_seed(0)
#         output = pipe('anime turle', generator=generator, output_type='np')
#         image = output.images[0]
#         assert image.shape == (768, 768, 3)
#         assert_mean_pixel_difference(image, expected_image)
