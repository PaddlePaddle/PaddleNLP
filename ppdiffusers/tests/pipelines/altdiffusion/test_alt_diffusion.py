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

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, XLMRobertaTokenizer
from ppdiffusers import (
    AltDiffusionPipeline,
    AutoencoderKL,
    DDIMScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from ppdiffusers.pipelines.alt_diffusion.modeling_roberta_series import (
    RobertaSeriesConfig,
    RobertaSeriesModelWithTransformation,
)
from ppdiffusers.utils import slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class AltDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AltDiffusionPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS

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
        )
        paddle.seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            projection_dim=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=5002,
        )
        text_encoder = CLIPTextModel(text_encoder_config).eval()
        tokenizer = XLMRobertaTokenizer.from_pretrained(
            "hf-internal-testing/tiny-xlm-roberta", model_max_length=77
        )  # must set model_max_length 77 here
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
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_alt_diffusion_ddim(self):
        components = self.get_dummy_components()
        paddle.seed(0)
        text_encoder_config = RobertaSeriesConfig(
            hidden_size=32,
            project_dim=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            vocab_size=5002,
        )
        text_encoder = RobertaSeriesModelWithTransformation(text_encoder_config).eval()
        components["text_encoder"] = text_encoder
        alt_pipe = AltDiffusionPipeline(**components)
        alt_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = "A photo of an astronaut"
        output = alt_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.32336113, 0.2371237, 0.34009337, 0.22972241, 0.23742735, 0.4925817, 0.22020563, 0.20505491, 0.43374813]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_alt_diffusion_pndm(self):
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        paddle.seed(0)
        text_encoder_config = RobertaSeriesConfig(
            hidden_size=32,
            project_dim=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            vocab_size=5002,
        )
        text_encoder = RobertaSeriesModelWithTransformation(text_encoder_config).eval()
        components["text_encoder"] = text_encoder
        alt_pipe = AltDiffusionPipeline(**components)
        alt_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = alt_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.24095133, 0.26875997, 0.34291863, 0.2529385, 0.2736602, 0.49928105, 0.23973131, 0.21133915, 0.41810605]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01


@slow
@require_paddle_gpu
class AltDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_alt_diffusion(self):
        alt_pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion", safety_checker=None)
        alt_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = alt_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=20, output_type="np")
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [
                0.8718514442443848,
                0.8715569972991943,
                0.8748429417610168,
                0.8708409070968628,
                0.8782679438591003,
                0.8931069374084473,
                0.883078932762146,
                0.881088376045227,
                0.8617547154426575,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_alt_diffusion_fast_ddim(self):
        scheduler = DDIMScheduler.from_pretrained("BAAI/AltDiffusion", subfolder="scheduler")
        alt_pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion", scheduler=scheduler, safety_checker=None)
        alt_pipe.set_progress_bar_config(disable=None)
        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        output = alt_pipe([prompt], generator=generator, num_inference_steps=2, output_type="numpy")
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [
                0.9265012741088867,
                0.9305188059806824,
                0.8999797105789185,
                0.9346827268600464,
                0.9264709949493408,
                0.9447494745254517,
                0.9428927898406982,
                0.9417785406112671,
                0.9157286882400513,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
