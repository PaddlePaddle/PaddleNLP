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

import random
import unittest

import paddle
from ppdiffusers_test.pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import (
    CLIPFeatureExtractor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableUnCLIPImg2ImgPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.pipelines.pipeline_utils import DiffusionPipeline
from ppdiffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import (
    StableUnCLIPImageNormalizer,
)
from ppdiffusers.utils.import_utils import is_ppxformers_available
from ppdiffusers.utils.testing_utils import floats_tensor


class StableUnCLIPImg2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableUnCLIPImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    def get_dummy_components(self):
        embedder_hidden_size = 32
        embedder_projection_dim = embedder_hidden_size
        feature_extractor = CLIPFeatureExtractor(crop_size=32, size=32)
        image_encoder = CLIPVisionModelWithProjection(
            CLIPVisionConfig(
                hidden_size=embedder_hidden_size,
                projection_dim=embedder_projection_dim,
                num_hidden_layers=5,
                num_attention_heads=4,
                image_size=32,
                intermediate_size=37,
                patch_size=1,
            )
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
            "feature_extractor": feature_extractor,
            "image_encoder": image_encoder,
            "image_normalizer": image_normalizer,
            "image_noising_scheduler": image_noising_scheduler,
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
        }
        return components

    def get_dummy_inputs(self, seed=0, pil_image=True):
        generator = paddle.Generator().manual_seed(seed)

        input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        if pil_image:
            input_image = input_image * 0.5 + 0.5
            input_image = input_image.clip(min=0, max=1)
            input_image = input_image.cpu().transpose(perm=[0, 2, 3, 1]).float().numpy()
            input_image = DiffusionPipeline.numpy_to_pil(input_image)[0]
        return {
            "prompt": "An anime racoon running a marathon",
            "image": input_image,
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
        }

    def test_attention_slicing_forward_pass(self):
        test_max_difference = False
        self._test_attention_slicing_forward_pass(test_max_difference=test_max_difference)

    def test_inference_batch_single_identical(self):
        test_max_difference = False
        self._test_inference_batch_single_identical(test_max_difference=test_max_difference)

    @unittest.skipIf(
        not is_ppxformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(test_max_difference=False)


# @slow
# @require_paddle_gpu
# class StableUnCLIPImg2ImgPipelineIntegrationTests(unittest.TestCase):

#     def tearDown(self):
#         super().tearDown()
#         gc.collect()
#         paddle.device.cuda.empty_cache()

#     def test_stable_unclip_l_img2img(self):
#         input_image = load_image(
#             'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/turtle.png'
#             )
#         expected_image = load_numpy(
#             'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/stable_unclip_2_1_l_img2img_anime_turtle_fp16.npy'
#             )
#         pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
#             'fusing/stable-unclip-2-1-l-img2img')
#         pipe.set_progress_bar_config(disable=None)
#         generator = paddle.Generator().manual_seed(0)
#         output = pipe('anime turle', image=input_image, generator=generator,
#             output_type='np')
#         image = output.images[0]
#         # breakpoint()
#         assert image.shape == (768, 768, 3)
#         assert_mean_pixel_difference(image, expected_image)

#     def test_stable_unclip_h_img2img(self):
#         input_image = load_image(
#             'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/turtle.png'
#             )
#         expected_image = load_numpy(
#             'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/stable_unclip_2_1_h_img2img_anime_turtle_fp16.npy'
#             )
#         pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
#             'fusing/stable-unclip-2-1-h-img2img')
#         pipe.set_progress_bar_config(disable=None)
#         generator = paddle.Generator().manual_seed(0)
#         output = pipe('anime turle', image=input_image, generator=generator,
#             output_type='np')
#         image = output.images[0]
#         assert image.shape == (768, 768, 3)
#         assert_mean_pixel_difference(image, expected_image)
