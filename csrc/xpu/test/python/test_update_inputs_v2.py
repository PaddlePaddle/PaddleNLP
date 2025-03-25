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

import unittest

import numpy as np
import paddle
from paddlenlp_ops import update_inputs_v2

np.random.seed(2023)


class GetUpdateInputsTest(unittest.TestCase):
    def test_update_inputs(self):

        tensor1 = paddle.to_tensor([[False]], dtype="bool", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor2 = paddle.to_tensor([[32]], dtype="int64", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor3 = paddle.to_tensor([True], dtype="bool", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor4 = paddle.to_tensor([[1]], dtype="int32", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor5 = paddle.to_tensor([[0]], dtype="int32", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor6 = paddle.to_tensor([[44]], dtype="int32", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor7 = paddle.to_tensor([[200]], dtype="int64", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor8 = paddle.to_tensor(
            [[2160, 5726, 25, 100001, 100001, 100001]], dtype="int64", place=paddle.XPUPlace(0), stop_gradient=True
        )
        tensor9 = paddle.to_tensor([1], dtype="int64", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor10 = paddle.to_tensor([[100001]], dtype="int64", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor11 = paddle.to_tensor([False], dtype="bool", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor12 = paddle.to_tensor([[100001]], dtype="int64", place=paddle.XPUPlace(0), stop_gradient=True)
        tensor13 = paddle.to_tensor([[2160]], dtype="int64", place=paddle.XPUPlace(0), stop_gradient=True)

        # 将 Tensors 合并到一个列表中
        arg = [
            tensor1,
            tensor2,
            tensor3,
            tensor4,
            tensor5,
            tensor6,
            tensor7,
            tensor8,
            tensor9,
            tensor10,
            tensor11,
            tensor12,
            tensor13,
        ]

        print(arg)
        print("---------------")

        update_inputs_v2(*arg)

        print(arg)

        assert tensor1 is True
        assert tensor3 is False


if __name__ == "__main__":
    unittest.main()
