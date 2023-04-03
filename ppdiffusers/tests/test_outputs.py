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
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from ppdiffusers.utils.outputs import BaseOutput


@dataclass
class CustomOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]


class ConfigTester(unittest.TestCase):
    def test_outputs_single_attribute(self):
        outputs = CustomOutput(images=np.random.rand(1, 3, 4, 4))
        assert isinstance(outputs.images, np.ndarray)
        assert outputs.images.shape == (1, 3, 4, 4)
        assert isinstance(outputs["images"], np.ndarray)
        assert outputs["images"].shape == (1, 3, 4, 4)
        assert isinstance(outputs[0], np.ndarray)
        assert outputs[0].shape == (1, 3, 4, 4)
        outputs = CustomOutput(images=[PIL.Image.new("RGB", (4, 4))])
        assert isinstance(outputs.images, list)
        assert isinstance(outputs.images[0], PIL.Image.Image)
        assert isinstance(outputs["images"], list)
        assert isinstance(outputs["images"][0], PIL.Image.Image)
        assert isinstance(outputs[0], list)
        assert isinstance(outputs[0][0], PIL.Image.Image)

    def test_outputs_dict_init(self):
        outputs = CustomOutput({"images": np.random.rand(1, 3, 4, 4)})
        assert isinstance(outputs.images, np.ndarray)
        assert outputs.images.shape == (1, 3, 4, 4)
        assert isinstance(outputs["images"], np.ndarray)
        assert outputs["images"].shape == (1, 3, 4, 4)
        assert isinstance(outputs[0], np.ndarray)
        assert outputs[0].shape == (1, 3, 4, 4)
        outputs = CustomOutput({"images": [PIL.Image.new("RGB", (4, 4))]})
        assert isinstance(outputs.images, list)
        assert isinstance(outputs.images[0], PIL.Image.Image)
        assert isinstance(outputs["images"], list)
        assert isinstance(outputs["images"][0], PIL.Image.Image)
        assert isinstance(outputs[0], list)
        assert isinstance(outputs[0][0], PIL.Image.Image)
