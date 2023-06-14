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

"""
annotator utils.
"""

import numpy as np
import paddle
from annotator.canny import CannyDetector
from annotator.hed import HEDdetector
from annotator.util import HWC3
from paddle.vision import transforms


def create_annotator(control_type):
    """create_annotator by control type."""
    if control_type == "canny":
        return CannyProcessor()
    elif control_type == "hed":
        return HedProcessor()
    elif control_type == "raw":
        return DummyProcessor()
    else:
        raise NotImplementedError


class DummyProcessor:
    """
    Dummy.
    """

    def __init__(self):
        self.post_process = transforms.ToTensor()

    def process_data_load(self, image):
        """
        Args:
          image: PIL image.
        Return:
          numpy or tensor. (0 ~ 1)
        """
        res = self.post_process(image)
        return res

    def process_model_forward(self, image):
        """dummy"""
        return image


class CannyProcessor:
    """
    canny wrapper.
    """

    def __init__(self):
        self.canny_thresh = (100, 200)
        self.apply_canny = CannyDetector()
        self.post_process = transforms.ToTensor()

    def process_data_load(self, image):
        """
        Args:
          image: PIL image.
        Return:
          numpy or tensor. (0 ~ 1)
        """
        image = np.array(image)
        img = HWC3(image)
        H, W, C = img.shape
        # TODO: random thresh.
        detected_map = self.apply_canny(img, *self.canny_thresh)
        detected_map = HWC3(detected_map)
        res = self.post_process(detected_map)
        return res

    def process_model_forward(self, image):
        """
        Args:
          tensor (GPU)
        Return:
          tensor (GPU)
        """
        return image


class HedProcessor:
    """
    HED wrapper.
    """

    def __init__(self):
        self.apply_hed = HEDdetector(modelpath="you/hed/model")
        self.post_process = transforms.ToTensor()

    def process_data_load(self, image):
        """
        Args:
          image: PIL image.
        Return:
          numpy or tensor.
        """
        image = np.array(image)
        img = HWC3(image)  # numpy shape=(H, W, C), RGB
        img = image[:, :, ::-1]  # numpy shape=(H, W, C), BGR
        res = self.post_process(img)  # tensor, shape=(C, H, W), BGR, \in (0, 1)
        return res

    def process_model_forward(self, image):
        """
        Args:
          tensor (GPU), shape=(B, 3, H, W), (0, 1)
        Return:
          tensor (GPU), shape=(B, 3, H, W), (0, 1)
        """
        with paddle.no_grad():
            edge = self.apply_hed.netNetwork(image)  # (B, 1, H, W)
            B, C, H, W = edge.shape
            edge = edge.expand([B, 3, H, W])  # (B, 3, H, W)
        return edge
