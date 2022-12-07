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

from ppdiffusers import DanceDiffusionPipeline, IPNDMScheduler, UNet1DModel
from ppdiffusers.utils import slow


class PipelineFastTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @property
    def dummy_unet(self):
        paddle.seed(0)
        model = UNet1DModel(
            block_out_channels=(32, 32, 64),
            extra_in_channels=16,
            sample_size=512,
            sample_rate=16_000,
            in_channels=2,
            out_channels=2,
            flip_sin_to_cos=True,
            use_timestep_embedding=False,
            time_embedding_type="fourier",
            mid_block_type="UNetMidBlock1D",
            down_block_types=["DownBlock1DNoSkip"] + ["DownBlock1D"] + ["AttnDownBlock1D"],
            up_block_types=["AttnUpBlock1D"] + ["UpBlock1D"] + ["UpBlock1DNoSkip"],
        )
        return model

    def test_dance_diffusion(self):
        scheduler = IPNDMScheduler()

        pipe = DanceDiffusionPipeline(unet=self.dummy_unet, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        output = pipe(generator=generator, num_inference_steps=4)
        audio = output.audios

        generator = paddle.Generator().manual_seed(0)
        output = pipe(generator=generator, num_inference_steps=4, return_dict=False)
        audio_from_tuple = output[0]

        audio_slice = audio[0, -3:, -3:]
        audio_from_tuple_slice = audio_from_tuple[0, -3:, -3:]

        assert audio.shape == (1, 2, self.dummy_unet.sample_size)
        expected_slice = np.array([0.3497878611087799, -0.10828632861375809, -1.0, -1.0, -1.0, 0.1466890275478363])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(audio_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
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
        expected_slice = np.array([-0.1576, -0.1526, -0.127, -0.2699, -0.2762, -0.2487])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 1e-2
