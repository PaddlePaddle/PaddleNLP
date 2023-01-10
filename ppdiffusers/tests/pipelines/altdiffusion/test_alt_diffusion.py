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
from test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import XLMRobertaTokenizer
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


class AltDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AltDiffusionPipeline

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=64,
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
        config = RobertaSeriesConfig(
            hidden_size=32,
            project_dim=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            vocab_size=5002,
            return_dict=True,
        )
        text_encoder = RobertaSeriesModelWithTransformation(config)
        text_encoder.eval()

        tokenizer = XLMRobertaTokenizer.from_pretrained("hf-internal-testing/tiny-xlm-roberta", model_max_length=77)

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
        # make sure here that pndm scheduler skips prk
        alt_pipe = AltDiffusionPipeline(**components)
        alt_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = "A photo of an astronaut"
        output = alt_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array(
            [
                0.758880615234375,
                0.8570979833602905,
                0.4303036034107208,
                0.4664992094039917,
                0.34632986783981323,
                0.2825028896331787,
                0.4622049927711487,
                0.4205690622329712,
                0.5209154486656189,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_alt_diffusion_pndm(self):
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)

        # make sure here that pndm scheduler skips prk
        alt_pipe = AltDiffusionPipeline(**components)
        alt_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        output = alt_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array(
            [
                0.8294100761413574,
                0.93553626537323,
                0.45852237939834595,
                0.4506404995918274,
                0.3515189290046692,
                0.2775656580924988,
                0.39653486013412476,
                0.38815104961395264,
                0.535513162612915,
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


@slow
class AltDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_alt_diffusion(self):
        # make sure here that pndm scheduler skips prk
        alt_pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion", safety_checker=None)
        alt_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)
        with paddle.amp.auto_cast(True):
            output = alt_pipe(
                [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=20, output_type="np"
            )

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.8720703, 0.87109375, 0.87402344, 0.87109375, 0.8779297, 0.8925781, 0.8823242, 0.8808594, 0.8613281]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_alt_diffusion_fast_ddim(self):
        scheduler = DDIMScheduler.from_pretrained("BAAI/AltDiffusion", subfolder="scheduler")

        alt_pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion", scheduler=scheduler, safety_checker=None)
        alt_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = paddle.Generator().manual_seed(0)

        with paddle.amp.auto_cast(True):
            output = alt_pipe([prompt], generator=generator, num_inference_steps=2, output_type="numpy")
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array(
            [0.9267578, 0.9301758, 0.9013672, 0.9345703, 0.92578125, 0.94433594, 0.9423828, 0.9423828, 0.9160156]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
