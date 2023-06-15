# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy

import cv2
import numpy as np
from annotator.util import annotator_ckpts_path
from einops import rearrange

from .api_inference import MidasInference


class MidasDetector_Infer:
    def __init__(self):
        self.model = MidasInference(annotator_ckpts_path)

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        assert input_image.ndim == 3
        image_depth = input_image
        image_depth = rearrange(image_depth, "h w c -> 1 c h w")
        image_depth = image_depth.astype("float32")
        image_depth = image_depth / 127.5 - 1.0
        depth = self.model.predict(image_depth)[0]
        depth_pt = copy.deepcopy(depth)
        depth_pt -= depth_pt.min()
        depth_pt /= depth_pt.max()
        depth_image = (depth_pt * 255.0).clip(min=0, max=255).astype(np.uint8)
        depth_np = depth
        x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
        y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
        z = np.ones_like(x) * a
        x[depth_pt < bg_th] = 0
        y[depth_pt < bg_th] = 0
        normal = np.stack([x, y, z], axis=2)
        normal /= np.sum(normal**2.0, axis=2, keepdims=True) ** 0.5
        normal_image = (normal * 127.5 + 127.5).clip(min=0, max=255).astype(np.uint8)
        return depth_image, normal_image
