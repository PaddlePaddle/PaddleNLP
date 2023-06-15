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
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusionLatentUpscalePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionLatentUpscalePipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {
        "height",
        "width",
        "cross_attention_kwargs",
        "negative_prompt_embeds",
        "prompt_embeds",
    }
    required_optional_params = PipelineTesterMixin.required_optional_params - {"num_images_per_prompt"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    test_cpu_offload = False

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 4
        sizes = 16, 16
        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
        return image

    def get_dummy_components(self):
        paddle.seed(0)
        model = UNet2DConditionModel(
            act_fn="gelu",
            attention_head_dim=8,
            norm_num_groups=None,
            block_out_channels=[32, 32, 64, 64],
            time_cond_proj_dim=160,
            conv_in_kernel=1,
            conv_out_kernel=1,
            cross_attention_dim=32,
            down_block_types=(
                "KDownBlock2D",
                "KCrossAttnDownBlock2D",
                "KCrossAttnDownBlock2D",
                "KCrossAttnDownBlock2D",
            ),
            in_channels=8,
            mid_block_type=None,
            only_cross_attention=False,
            out_channels=5,
            resnet_time_scale_shift="scale_shift",
            time_embedding_type="fourier",
            timestep_post_act="gelu",
            up_block_types=("KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KUpBlock2D"),
        )
        vae = AutoencoderKL(
            block_out_channels=[32, 32, 64, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        scheduler = EulerDiscreteScheduler(prediction_type="sample")
        text_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="quick_gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_config).eval()
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        components = {
            "unet": model.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": self.dummy_image.cpu(),
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "numpy",
        }
        return inputs

    def test_inference(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        self.assertEqual(image.shape, (1, 256, 256, 3))
        expected_slice = np.array(
            [
                0.47222412,
                0.41921633,
                0.44717434,
                0.46874192,
                0.42588258,
                0.46150726,
                0.4677534,
                0.45583832,
                0.48579055,
            ]  # TODO check this
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 0.001)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(relax_max_difference=False)


@require_paddle_gpu
@slow
class StableDiffusionLatentUpscalePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_latent_upscaler_fp16(self):
        generator = paddle.Generator().manual_seed(seed=33)
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", paddle_dtype=paddle.float16)
        pipe.to("gpu")
        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", paddle_dtype=paddle.float16
        )
        upscaler.to("gpu")

        prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"
        low_res_latents = pipe(prompt, generator=generator, output_type="latent").images
        image = upscaler(
            prompt=prompt,
            image=low_res_latents,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
            output_type="np",
        ).images[0]
        # invalid expected_image
        # expected_image = load_numpy(
        #     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/astronaut_1024.npy"
        # )
        image = image[-3:, -3:, 0]
        expected_image = [
            [[0.03686523], [0.03759766], [0.05175781]],
            [[0.03491211], [0.05126953], [0.04541016]],
            [[0.02880859], [0.03369141], [0.05004883]],
        ]
        assert np.abs((expected_image - image).max()) < 0.5

    def test_latent_upscaler_fp16_image(self):
        generator = paddle.Generator().manual_seed(seed=33)
        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", paddle_dtype=paddle.float16
        )
        upscaler.to("gpu")

        prompt = "the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas"
        low_res_img = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/fire_temple_512.png"
        )
        image = upscaler(
            prompt=prompt,
            image=low_res_img,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
            output_type="np",
        ).images[0]
        # invalid expected_image
        # expected_image = load_numpy(
        #     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/fire_temple_1024.npy"
        # )
        image = image[-3:, -3:, 0]
        expected_image = [
            [[0.03686523], [0.03759766], [0.05151367]],
            [[0.03491211], [0.05151367], [0.04541016]],
            [[0.02880859], [0.03393555], [0.05004883]],
        ]
        assert np.abs((expected_image - image).max()) < 0.05
