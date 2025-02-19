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
import paddle
from safetensors.numpy import load_file, save_file

from paddlenlp.utils.safetensors import fast_load_file, fast_safe_open

from ..testing_utils import skip_platform

paddle.set_device("cpu")


def enhanced_to_tensor(tensor):
    if tensor.dtype == np.bfloat16:
        return paddle.to_tensor(tensor.view(np.uint16))
    if tensor.dtype == np.float8_e5m2:
        t = paddle.to_tensor(tensor.view(np.int8))
        new_t = paddle.empty(t.shape, dtype=paddle.float8_e5m2)
        new_t.get_tensor()._share_data_with(t.get_tensor())
        return new_t
    if tensor.dtype == np.float8_e4m3fn:
        t = paddle.to_tensor(tensor.view(np.int8))
        new_t = paddle.empty(t.shape, dtype=paddle.float8_e4m3fn)
        new_t.get_tensor()._share_data_with(t.get_tensor())
        return new_t
        # return paddle.to_tensor(tensor.view(np.int8), dtype=paddle.float8_e4m3fn)
    raise ValueError()
    return paddle.to_tensor(tensor)


class EextendDtypeNumpySafe(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.weight_map = {}
        self.tensors = [
            ([10, 1, 10], "float32"),
            ([1, 1, 10], "float32"),
            ([1, 1, 1, 10], "float32"),
            ([10, 10], "float32"),
            ([8], "float16"),
            ([5, 5, 5], "int32"),
        ]

    def get_target_dtype(self, dtype="float32"):
        count = 0
        weight_map = {}
        for shape, _ in self.tensors:
            weight_map[f"weight_{count}"] = (np.random.random(shape) * 100).astype(dtype)
            count += 1
        return weight_map

    def get_paddle_target_dtype(self, dtype="float32"):
        weight_map = self.get_target_dtype(dtype)
        for k, v in list(weight_map.items()):
            weight_map[k] = enhanced_to_tensor(v)
        return weight_map

    @skip_platform("win32", "cygwin")
    def test_save_load_file_paddle(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for dtype in ["bfloat16", "float8_e5m2", "float8_e4m3fn"]:
                weight_map = self.get_paddle_target_dtype(dtype)
                path = os.path.join(tmpdirname, "test.safetensors")
                shard = {}
                for k in list(weight_map.keys()):
                    if isinstance(weight_map[k], paddle.Tensor):
                        shard[k] = weight_map[k].cpu().numpy()
                    else:
                        shard[k] = weight_map[k]

                save_file(shard, path, metadata={"format": "np"})
                sf_load = load_file(path)
                fs_sf_load = fast_load_file(path)

                for k, v in self.weight_map.items():
                    paddle.allclose(v, enhanced_to_tensor(sf_load[k]))
                    paddle.allclose(v, enhanced_to_tensor(fs_sf_load[k]))

    @skip_platform("win32", "cygwin")
    def test_save_load_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for dtype in ["bfloat16", "float8_e4m3fn", "float8_e5m2"]:
                weight_map = self.get_target_dtype(dtype)
                path = os.path.join(tmpdirname, "test.safetensors")
                save_file(weight_map, path, metadata={"format": "np"})
                sf_load = load_file(path)
                fs_sf_load = fast_load_file(path)
                for k, v in self.weight_map.items():
                    np.testing.assert_equal(v, sf_load[k])
                    np.testing.assert_equal(v, fs_sf_load[k])

    @skip_platform("win32", "cygwin")
    def test_dtype_safe_open(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for dtype in ["float32", "int32", "bfloat16", "float8_e4m3fn", "float8_e5m2"]:
                weight_map = self.get_target_dtype(dtype)
                path = os.path.join(tmpdirname, "test.safetensors")
                save_file(weight_map, path, metadata={"format": "np"})

                with fast_safe_open(path, framework="np") as f:
                    for key in f.keys():
                        safe_slice = f.get_slice(key)
                        # np.testing.assert_equal(self.weight_map[key][2:1, ...], safe_slice[2:1, ...])
                        np.testing.assert_equal(weight_map[key][0, ...], safe_slice[0, ...])
                        np.testing.assert_equal(weight_map[key][0:1, ...], safe_slice[0:1, ...])
                        np.testing.assert_equal(weight_map[key][..., 2:], safe_slice[..., 2:])
                        np.testing.assert_equal(weight_map[key][..., 1], safe_slice[..., 1])
                        np.testing.assert_equal(weight_map[key][:2, ...], safe_slice[:2, ...])
                        np.testing.assert_equal(weight_map[key][..., :4], safe_slice[..., :4])


class FastSafetensors(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.weight_map = {}
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
            self.weight_map[f"weight_{count}"] = (np.random.random(shape) * 100).astype(dtype)
            count += 1

    @skip_platform("win32", "cygwin")
    def test_load_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "test.safetensors")
            save_file(self.weight_map, path, metadata={"format": "np"})
            sf_load = load_file(path)
            fs_sf_load = fast_load_file(path)
            for k, v in self.weight_map.items():
                np.testing.assert_equal(v, sf_load[k])
                np.testing.assert_equal(v, fs_sf_load[k])

    @skip_platform("win32", "cygwin")
    def test_safe_open(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "test.safetensors")
            save_file(self.weight_map, path, metadata={"format": "np"})

            with fast_safe_open(path, framework="np") as f:
                for key in f.keys():
                    safe_slice = f.get_slice(key)
                    # np.testing.assert_equal(self.weight_map[key][2:1, ...], safe_slice[2:1, ...])
                    np.testing.assert_equal(self.weight_map[key][0, ...], safe_slice[0, ...])
                    np.testing.assert_equal(self.weight_map[key][0:1, ...], safe_slice[0:1, ...])
                    np.testing.assert_equal(self.weight_map[key][..., 2:], safe_slice[..., 2:])
                    np.testing.assert_equal(self.weight_map[key][..., 1], safe_slice[..., 1])
                    np.testing.assert_equal(self.weight_map[key][:2, ...], safe_slice[:2, ...])
                    np.testing.assert_equal(self.weight_map[key][..., :4], safe_slice[..., :4])
