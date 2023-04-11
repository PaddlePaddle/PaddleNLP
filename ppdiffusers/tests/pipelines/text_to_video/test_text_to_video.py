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
    DPMSolverMultistepScheduler,
    TextToVideoSDPipeline,
    UNet3DConditionModel,
)
from ppdiffusers.utils import load_numpy, slow


class TextToVideoSDPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = TextToVideoSDPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    required_optional_params = frozenset(
        ["num_inference_steps", "generator", "latents", "return_dict", "callback", "callback_steps"]
    )

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet3DConditionModel(
            block_out_channels=(32, 64, 64, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
            cross_attention_dim=32,
            attention_head_dim=4,
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
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(0)
        # "output_type": "pd" is problematic
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "pd",
        }
        return inputs

    def test_text_to_video_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = TextToVideoSDPipeline(**components)
        sd_pipe = sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "np"
        frames = sd_pipe(**inputs).frames
        image_slice = frames[0][-3:, -3:, (-1)]
        assert frames[0].shape == (64, 64, 3)
        expected_slice = np.array([65, 138, 97, 105, 157, 113, 78, 111, 69])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    # def test_attention_slicing_forward_pass(self):
    #     self._test_attention_slicing_forward_pass(test_mean_pixel_difference=False)

    @unittest.skip(reason="Batching needs to be properly figured out first for this pipeline.")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip(reason="Batching needs to be properly figured out first for this pipeline.")
    def test_inference_batch_single_identical(self):
        pass

    @unittest.skip(reason="`num_images_per_prompt` argument is not supported for this pipeline.")
    def test_num_images_per_prompt(self):
        pass


@slow
class TextToVideoSDPipelineSlowTests(unittest.TestCase):
    def test_full_model(self):
        expected_video = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_to_video/video.npy"
        )
        pipe = TextToVideoSDPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b", from_hf_hub=True, from_diffusers=True
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe
        prompt = "Spiderman is surfing"
        generator = paddle.Generator().manual_seed(0)
        video_frames = pipe(prompt, generator=generator, num_inference_steps=25, output_type="pd").frames
        video = video_frames.cpu().numpy()
        assert np.abs(expected_video - video).mean() < 0.8

    def test_two_step_model(self):
        expected_video = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_to_video/video_2step.npy"
        )
        pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b")
        pipe = pipe
        prompt = "Spiderman is surfing"
        generator = paddle.Generator().manual_seed(0)
        video_frames = pipe(prompt, generator=generator, num_inference_steps=2, output_type="pd").frames
        video = video_frames.cpu().numpy()
        assert np.abs(expected_video - video).mean() < 0.8
