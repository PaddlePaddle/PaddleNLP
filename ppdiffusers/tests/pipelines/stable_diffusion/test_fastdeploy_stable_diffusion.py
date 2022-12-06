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

import unittest

import numpy as np
from test_pipelines_fastdeploy_common import FastDeployPipelineTesterMixin

from ppdiffusers import (
    DDIMScheduler,
    FastDeployStableDiffusionPipeline,
    LMSDiscreteScheduler,
)
from ppdiffusers.utils.testing_utils import (
    is_fastdeploy_available,
    require_fastdeploy,
    slow,
)

if is_fastdeploy_available():
    import fastdeploy as fd


class FastDeployStableDiffusionPipelineFastTests(FastDeployPipelineTesterMixin, unittest.TestCase):
    # FIXME: add fast tests
    pass


def create_runtime_option(device_id=-1, backend="paddle"):
    option = fd.RuntimeOption()
    if backend == "paddle":
        option.use_paddle_backend()
    else:
        option.use_ort_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
    return option


@slow
@require_fastdeploy
class FastDeployStableDiffusionPipelineIntegrationTests(unittest.TestCase):
    @property
    def runtime_options(self):
        return {
            "text_encoder": create_runtime_option(0, "onnx"),  # use gpu
            "vae_encoder": create_runtime_option(0, "paddle"),  # use gpu
            "vae_decoder": create_runtime_option(0, "paddle"),  # use gpu
            "unet": create_runtime_option(0, "paddle"),  # use gpu
        }

    def test_inference_default_pndm(self):
        # using the PNDM scheduler by default
        sd_pipe = FastDeployStableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4@fastdeploy",
            runtime_options=self.runtime_options,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        np.random.seed(0)
        output = sd_pipe([prompt], guidance_scale=6.0, num_inference_steps=10, output_type="np")
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0452, 0.0390, 0.0087, 0.0350, 0.0617, 0.0364, 0.0544, 0.0523, 0.0720])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_inference_ddim(self):
        ddim_scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5@fastdeploy", subfolder="scheduler"
        )
        sd_pipe = FastDeployStableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5@fastdeploy",
            scheduler=ddim_scheduler,
            runtime_options=self.runtime_options,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "open neural network exchange"
        generator = np.random.RandomState(0)
        output = sd_pipe([prompt], guidance_scale=7.5, num_inference_steps=10, generator=generator, output_type="np")
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.2867, 0.1974, 0.1481, 0.7294, 0.7251, 0.6667, 0.4194, 0.5642, 0.6486])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_inference_k_lms(self):
        lms_scheduler = LMSDiscreteScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5@fastdeploy", subfolder="scheduler"
        )
        sd_pipe = FastDeployStableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5@fastdeploy",
            scheduler=lms_scheduler,
            runtime_options=self.runtime_options,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "open neural network exchange"
        generator = np.random.RandomState(0)
        output = sd_pipe([prompt], guidance_scale=7.5, num_inference_steps=10, generator=generator, output_type="np")
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.2306, 0.1959, 0.1593, 0.6549, 0.6394, 0.5408, 0.5065, 0.6010, 0.6161])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_intermediate_state(self):
        number_of_steps = 0

        def test_callback_fn(step: int, timestep: int, latents: np.ndarray) -> None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.6772, -0.3835, -1.2456, 0.1905, -1.0974, 0.6967, -1.9353, 0.0178, 1.0167]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3
            elif step == 5:
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.3351, 0.2241, -0.1837, -0.2325, -0.6577, 0.3393, -0.0241, 0.5899, 1.3875]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 1e-3

        test_callback_fn.has_been_called = False

        pipe = FastDeployStableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5@fastdeploy",
            runtime_options=self.runtime_options,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "Andromeda galaxy in a bottle"

        generator = np.random.RandomState(0)
        pipe(
            prompt=prompt,
            num_inference_steps=5,
            guidance_scale=7.5,
            generator=generator,
            callback=test_callback_fn,
            callback_steps=1,
        )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 6
