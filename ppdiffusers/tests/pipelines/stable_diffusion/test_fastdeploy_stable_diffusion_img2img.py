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

from ppdiffusers import FastDeployStableDiffusionImg2ImgPipeline, LMSDiscreteScheduler
from ppdiffusers.utils.testing_utils import (
    is_fastdeploy_available,
    load_image,
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
class FastDeployStableDiffusionImg2ImgPipelineIntegrationTests(unittest.TestCase):
    @property
    def runtime_options(self):
        return {
            "text_encoder": create_runtime_option(0, "onnx"),  # use gpu
            "vae_encoder": create_runtime_option(0, "paddle"),  # use gpu
            "vae_decoder": create_runtime_option(0, "paddle"),  # use gpu
            "unet": create_runtime_option(0, "paddle"),  # use gpu
        }

    def test_inference_default_pndm(self):
        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))
        # using the PNDM scheduler by default
        pipe = FastDeployStableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4@fastdeploy",
            runtime_options=self.runtime_options,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A fantasy landscape, trending on artstation"

        generator = np.random.RandomState(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=10,
            generator=generator,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 383:386, -1]

        assert images.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.4909, 0.5059, 0.5372, 0.4623, 0.4876, 0.5049, 0.4820, 0.4956, 0.5019])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 2e-2

    def test_inference_k_lms(self):
        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/sketch-mountains-input.jpg"
        )
        init_image = init_image.resize((768, 512))
        lms_scheduler = LMSDiscreteScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5@fastdeploy", subfolder="scheduler"
        )
        pipe = FastDeployStableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5@fastdeploy",
            runtime_options=self.runtime_options,
            scheduler=lms_scheduler,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A fantasy landscape, trending on artstation"

        generator = np.random.RandomState(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=10,
            generator=generator,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 383:386, -1]

        assert images.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.7950, 0.7923, 0.7903, 0.5516, 0.5501, 0.5476, 0.4965, 0.4933, 0.4910])
        # TODO: lower the tolerance after finding the cause of FastDeployruntime reproducibility issues
        assert np.abs(image_slice.flatten() - expected_slice).max() < 2e-2
