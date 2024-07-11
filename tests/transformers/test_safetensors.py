# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load_file, save_file

from paddlenlp.utils.safetensors import fast_load_file, fast_safe_open

from ..testing_utils import skip_platform


class FastSafetensors(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.weigth_map = {}
        tensors = [
            ([10, 1, 10], "float32"),
            ([1, 1, 10], "float32"),
            ([1, 1, 1, 10], "float32"),
            ([10, 10], "float32"),
            ([8], "float16"),
            ([5, 5, 5], "int32"),
        ]
        count = 0
        for shape, dtype in tensors:
            self.weigth_map[f"weight_{count}"] = (np.random.random(shape) * 100).astype(dtype)
            count += 1
        print(self.weigth_map)

    @skip_platform("win32", "cygwin")
    def test_load_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "test.safetensors")
            save_file(self.weigth_map, path, metadata={"format": "np"})
            sf_load = load_file(path)
            fs_sf_load = fast_load_file(path)
            for k, v in self.weigth_map.items():
                np.testing.assert_equal(v, sf_load[k])
                np.testing.assert_equal(v, fs_sf_load[k])

    @skip_platform("win32", "cygwin")
    def test_safe_open(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "test.safetensors")
            save_file(self.weigth_map, path, metadata={"format": "np"})

            with fast_safe_open(path, framework="np") as f:
                for key in f.keys():
                    safe_slice = f.get_slice(key)
                    # np.testing.assert_equal(self.weigth_map[key][2:1, ...], safe_slice[2:1, ...])
                    np.testing.assert_equal(self.weigth_map[key][0, ...], safe_slice[0, ...])
                    np.testing.assert_equal(self.weigth_map[key][0:1, ...], safe_slice[0:1, ...])
                    np.testing.assert_equal(self.weigth_map[key][..., 2:], safe_slice[..., 2:])
                    np.testing.assert_equal(self.weigth_map[key][..., 1], safe_slice[..., 1])
                    np.testing.assert_equal(self.weigth_map[key][:2, ...], safe_slice[:2, ...])
                    np.testing.assert_equal(self.weigth_map[key][..., :4], safe_slice[..., :4])
