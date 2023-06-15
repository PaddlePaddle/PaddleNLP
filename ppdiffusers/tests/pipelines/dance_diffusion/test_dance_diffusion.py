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
    UNCONDITIONAL_AUDIO_GENERATION_BATCH_PARAMS,
    UNCONDITIONAL_AUDIO_GENERATION_PARAMS,
)
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from ppdiffusers import DanceDiffusionPipeline, IPNDMScheduler, UNet1DModel
from ppdiffusers.utils import slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class DanceDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DanceDiffusionPipeline
    test_attention_slicing = False
    test_cpu_offload = False
    params = UNCONDITIONAL_AUDIO_GENERATION_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "callback",
        "latents",
        "callback_steps",
        "output_type",
        "num_images_per_prompt",
    }
    batch_params = UNCONDITIONAL_AUDIO_GENERATION_BATCH_PARAMS

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet1DModel(
            block_out_channels=(32, 32, 64),
            extra_in_channels=16,
            sample_size=512,
            sample_rate=16000,
            in_channels=2,
            out_channels=2,
            flip_sin_to_cos=True,
            use_timestep_embedding=False,
            time_embedding_type="fourier",
            mid_block_type="UNetMidBlock1D",
            down_block_types=("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
        )
        scheduler = IPNDMScheduler()
        components = {"unet": unet, "scheduler": scheduler}
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        inputs = {"batch_size": 1, "generator": generator, "num_inference_steps": 4}
        return inputs

    def test_dance_diffusion(self):
        components = self.get_dummy_components()
        pipe = DanceDiffusionPipeline(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)
        audio = output.audios
        audio_slice = audio[0, -3:, -3:]
        assert audio.shape == (1, 2, components["unet"].sample_size)
        expected_slice = np.array([1.0, 1.0, 0.9972942, -0.4477799, -0.5952974, 1.0])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 0.01


@slow
@require_paddle_gpu
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_dance_diffusion(self):
        pipe = DanceDiffusionPipeline.from_pretrained("harmonai/maestro-150k")
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        output = pipe(generator=generator, num_inference_steps=100, audio_length_in_s=4.096)
        audio = output.audios
        audio_slice = audio[0, -3:, -3:]
        assert audio.shape == (1, 2, pipe.unet.sample_size)
        expected_slice = np.array([-0.15758808, -0.15257765, -0.12701476, -0.26994032, -0.27616554, -0.24865153])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 0.01

    def test_dance_diffusion_fp16(self):
        pipe = DanceDiffusionPipeline.from_pretrained("harmonai/maestro-150k", paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        output = pipe(generator=generator, num_inference_steps=100, audio_length_in_s=4.096)
        audio = output.audios
        audio_slice = audio[0, -3:, -3:]
        assert audio.shape == (1, 2, pipe.unet.sample_size)
        # scheduler use fp32
        expected_slice = np.array([-0.15350387, -0.14624646, -0.12091318, -0.25969276, -0.26154587, -0.23359495])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 0.01
