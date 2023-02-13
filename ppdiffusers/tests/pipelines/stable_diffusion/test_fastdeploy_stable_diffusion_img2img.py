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

import random
import unittest

import numpy as np
from test_pipelines_fastdeploy_common import FastDeployPipelineTesterMixin

from ppdiffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FastDeployStableDiffusionImg2ImgPipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ppdiffusers.utils import floats_tensor
from ppdiffusers.utils.testing_utils import (
    is_fastdeploy_available,
    load_image,
    nightly,
    require_fastdeploy,
    slow,
)

if is_fastdeploy_available():
    import fastdeploy as fd


@require_fastdeploy
class FastDeployStableDiffusionPipelineFastTests(FastDeployPipelineTesterMixin, unittest.TestCase):
    hub_checkpoint = "hf-internal-testing/tiny-random-FastDeployStableDiffusionPipeline"

    @property
    def runtime_options(self):
        return {
            "text_encoder": create_runtime_option(0, "onnx"),  # use gpu
            "vae_encoder": create_runtime_option(0, "paddle"),  # use gpu
            "vae_decoder": create_runtime_option(0, "paddle"),  # use gpu
            "unet": create_runtime_option(0, "paddle"),  # use gpu
        }

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 128, 128), rng=random.Random(seed))
        generator = np.random.RandomState(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_pipeline_default_ddim(self):
        pipe = FastDeployStableDiffusionImg2ImgPipeline.from_pretrained(
            self.hub_checkpoint,
            runtime_options=self.runtime_options,
        )
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.69643, 0.58484, 0.50314, 0.58760, 0.55368, 0.59643, 0.51529, 0.41217, 0.49087])
        assert np.abs(image_slice - expected_slice).max() < 1e-1

    def test_pipeline_pndm(self):
        pipe = FastDeployStableDiffusionImg2ImgPipeline.from_pretrained(
            self.hub_checkpoint,
            runtime_options=self.runtime_options,
        )
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config, skip_prk_steps=True)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.61710, 0.53390, 0.49310, 0.55622, 0.50982, 0.58240, 0.50716, 0.38629, 0.46856])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_lms(self):
        pipe = FastDeployStableDiffusionImg2ImgPipeline.from_pretrained(
            self.hub_checkpoint,
            runtime_options=self.runtime_options,
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        # warmup pass to apply optimizations
        _ = pipe(**self.get_dummy_inputs())

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.52761, 0.59977, 0.49033, 0.49619, 0.54282, 0.50311, 0.47600, 0.40918, 0.45203])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_euler(self):
        pipe = FastDeployStableDiffusionImg2ImgPipeline.from_pretrained(
            self.hub_checkpoint,
            runtime_options=self.runtime_options,
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.52911, 0.60004, 0.49229, 0.49805, 0.54502, 0.50680, 0.47777, 0.41028, 0.45304])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_euler_ancestral(self):
        pipe = FastDeployStableDiffusionImg2ImgPipeline.from_pretrained(
            self.hub_checkpoint,
            runtime_options=self.runtime_options,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.52911, 0.60004, 0.49229, 0.49805, 0.54502, 0.50680, 0.47777, 0.41028, 0.45304])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1

    def test_pipeline_dpm_multistep(self):
        pipe = FastDeployStableDiffusionImg2ImgPipeline.from_pretrained(
            self.hub_checkpoint,
            runtime_options=self.runtime_options,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)
        expected_slice = np.array([0.65331, 0.58277, 0.48204, 0.56059, 0.53665, 0.56235, 0.50969, 0.40009, 0.46552])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-1


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


@nightly
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
            num_inference_steps=20,
            generator=generator,
            output_type="np",
        )
        images = output.images
        image_slice = images[0, 255:258, 383:386, -1]

        assert images.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.8043, 0.926, 0.9581, 0.8119, 0.8954, 0.913, 0.7209, 0.7463, 0.7431])
        # TODO: lower the tolerance after finding the cause of FastDeployruntime reproducibility issues
        assert np.abs(image_slice.flatten() - expected_slice).max() < 2e-2
