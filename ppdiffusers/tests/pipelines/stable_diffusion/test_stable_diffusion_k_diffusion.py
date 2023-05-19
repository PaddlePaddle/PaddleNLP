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

# import gc
# import unittest

# import numpy as np
# import paddle

# from ppdiffusers import StableDiffusionKDiffusionPipeline
# from ppdiffusers.utils import slow
# from ppdiffusers.utils.testing_utils import require_paddle_gpu


# @slow
# @require_paddle_gpu
# class StableDiffusionPipelineIntegrationTests(unittest.TestCase):
#     def tearDown(self):
#         super().tearDown()
#         gc.collect()
#         paddle.device.cuda.empty_cache()

#     def test_stable_diffusion_1(self):
#         sd_pipe = StableDiffusionKDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
#         sd_pipe.set_progress_bar_config(disable=None)
#         sd_pipe.set_scheduler("sample_euler")
#         prompt = "A painting of a squirrel eating a burger"
#         generator = paddle.Generator().manual_seed(0)
#         output = sd_pipe([prompt], generator=generator, guidance_scale=9.0, num_inference_steps=20, output_type="np")
#         image = output.images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 512, 512, 3)
#         expected_slice = np.array([0.0447, 0.0492, 0.0468, 0.0408, 0.0383, 0.0408, 0.0354, 0.038, 0.0339])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

#     def test_stable_diffusion_2(self):
#         sd_pipe = StableDiffusionKDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
#         sd_pipe.set_progress_bar_config(disable=None)
#         sd_pipe.set_scheduler("sample_euler")
#         prompt = "A painting of a squirrel eating a burger"
#         generator = paddle.Generator().manual_seed(0)
#         output = sd_pipe([prompt], generator=generator, guidance_scale=9.0, num_inference_steps=20, output_type="np")
#         image = output.images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 512, 512, 3)
#         expected_slice = np.array([0.1237, 0.132, 0.1438, 0.1359, 0.139, 0.1132, 0.1277, 0.1175, 0.1112])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.5
