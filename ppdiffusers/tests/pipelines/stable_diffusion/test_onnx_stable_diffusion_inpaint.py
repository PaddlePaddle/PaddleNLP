# # Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import unittest

# import numpy as np

# from ppdiffusers import LMSDiscreteScheduler, OnnxStableDiffusionInpaintPipeline
# from ppdiffusers.utils.testing_utils import (
#     is_onnx_available,
#     load_image,
#     nightly,
#     require_onnxruntime,
#     require_paddle_gpu,
# )

# from ...test_pipelines_onnx_common import OnnxPipelineTesterMixin

# if is_onnx_available():
#     import onnxruntime as ort


# class OnnxStableDiffusionPipelineFastTests(OnnxPipelineTesterMixin, unittest.TestCase):
#     pass


# @nightly
# @require_onnxruntime
# @require_paddle_gpu
# class OnnxStableDiffusionInpaintPipelineIntegrationTests(unittest.TestCase):
#     @property
#     def gpu_provider(self):
#         return "CUDAExecutionProvider", {"gpu_mem_limit": "15000000000", "arena_extend_strategy": "kSameAsRequested"}

#     @property
#     def gpu_options(self):
#         options = ort.SessionOptions()
#         options.enable_mem_pattern = False
#         return options

#     def test_inference_default_pndm(self):
#         init_image = load_image(
#             "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo.png"
#         )
#         mask_image = load_image(
#             "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
#         )
#         pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
#             "runwayml/stable-diffusion-inpainting",
#             revision="onnx",
#             safety_checker=None,
#             feature_extractor=None,
#             provider=self.gpu_provider,
#             sess_options=self.gpu_options,
#         )
#         pipe.set_progress_bar_config(disable=None)
#         prompt = "A red cat sitting on a park bench"
#         generator = np.random.RandomState(0)
#         output = pipe(
#             prompt=prompt,
#             image=init_image,
#             mask_image=mask_image,
#             guidance_scale=7.5,
#             num_inference_steps=10,
#             generator=generator,
#             output_type="np",
#         )
#         images = output.images
#         image_slice = images[0, 255:258, 255:258, -1]
#         assert images.shape == (1, 512, 512, 3)
#         expected_slice = np.array([0.2514, 0.3007, 0.3517, 0.179, 0.2382, 0.3167, 0.1944, 0.2273, 0.2464])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

#     def test_inference_k_lms(self):
#         init_image = load_image(
#             "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo.png"
#         )
#         mask_image = load_image(
#             "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
#         )
#         lms_scheduler = LMSDiscreteScheduler.from_pretrained(
#             "runwayml/stable-diffusion-inpainting", subfolder="scheduler", revision="onnx"
#         )
#         pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
#             "runwayml/stable-diffusion-inpainting",
#             revision="onnx",
#             scheduler=lms_scheduler,
#             safety_checker=None,
#             feature_extractor=None,
#             provider=self.gpu_provider,
#             sess_options=self.gpu_options,
#         )
#         pipe.set_progress_bar_config(disable=None)
#         prompt = "A red cat sitting on a park bench"
#         generator = np.random.RandomState(0)
#         output = pipe(
#             prompt=prompt,
#             image=init_image,
#             mask_image=mask_image,
#             guidance_scale=7.5,
#             num_inference_steps=20,
#             generator=generator,
#             output_type="np",
#         )
#         images = output.images
#         image_slice = images[0, 255:258, 255:258, -1]
#         assert images.shape == (1, 512, 512, 3)
#         expected_slice = np.array([0.0086, 0.0077, 0.0083, 0.0093, 0.0107, 0.0139, 0.0094, 0.0097, 0.0125])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
