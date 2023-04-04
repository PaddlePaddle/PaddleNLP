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

# from ppdiffusers import OnnxStableDiffusionInpaintPipelineLegacy
# from ppdiffusers.utils.testing_utils import (
#     is_onnx_available,
#     load_image,
#     load_numpy,
#     nightly,
#     require_onnxruntime,
#     require_paddle_gpu,
# )

# if is_onnx_available():
#     import onnxruntime as ort


# @nightly
# @require_onnxruntime
# @require_paddle_gpu
# class StableDiffusionOnnxInpaintLegacyPipelineIntegrationTests(unittest.TestCase):
#     @property
#     def gpu_provider(self):
#         return "CUDAExecutionProvider", {"gpu_mem_limit": "15000000000", "arena_extend_strategy": "kSameAsRequested"}

#     @property
#     def gpu_options(self):
#         options = ort.SessionOptions()
#         options.enable_mem_pattern = False
#         return options

#     def test_inference(self):
#         init_image = load_image(
#             "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo.png"
#         )
#         mask_image = load_image(
#             "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
#         )
#         expected_image = load_numpy(
#             "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/red_cat_sitting_on_a_park_bench_onnx.npy"
#         )
#         pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
#             "CompVis/stable-diffusion-v1-4",
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
#             strength=0.75,
#             guidance_scale=7.5,
#             num_inference_steps=15,
#             generator=generator,
#             output_type="np",
#         )
#         image = output.images[0]
#         assert image.shape == (512, 512, 3)
#         assert np.abs(expected_image - image).max() < 0.01
