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
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionAttendAndExcitePipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import load_numpy, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusionAttendAndExcitePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionAttendAndExcitePipeline
    test_attention_slicing = False
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS.union({"token_indices"})

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            attention_head_dim=(2, 4),
            use_linear_projection=True,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        paddle.seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        paddle.seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_encoder_config).eval()
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        inputs = inputs = {
            "prompt": "a cat and a frog",
            "token_indices": [2, 5],
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "max_iter_to_alter": 2,
            "thresholds": {(0): 0.7},
        }
        return inputs

    def test_inference(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        self.assertEqual(image.shape, (1, 64, 64, 3))
        expected_slice = np.array(
            [
                0.14484110474586487,
                0.16710084676742554,
                0.3591206669807434,
                0.30512064695358276,
                0.2687497138977051,
                0.4805506765842438,
                0.20869559049606323,
                0.12253415584564209,
                0.3426440954208374,
            ]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 0.001)

    def test_inference_batch_consistent(self):
        # NOTE: Larger batch sizes cause this test to timeout, only test on smaller batches
        self._test_inference_batch_consistent(batch_sizes=[2, 4])


@require_paddle_gpu
@slow
class StableDiffusionAttendAndExcitePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_attend_and_excite_fp16(self):
        generator = paddle.Generator().manual_seed(seed=51)
        pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, paddle_dtype=paddle.float16
        )

        prompt = "a painting of an elephant with glasses"
        token_indices = [5, 7]
        image = pipe(
            prompt=prompt,
            token_indices=token_indices,
            guidance_scale=7.5,
            generator=generator,
            num_inference_steps=5,
            max_iter_to_alter=5,
            output_type="numpy",
        ).images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/attend-and-excite/elephant_glasses.npy"
        )
        assert np.abs((expected_image - image).max()) < 0.5
