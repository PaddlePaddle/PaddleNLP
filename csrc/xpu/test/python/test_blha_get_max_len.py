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

np.random.seed(2023)


class GetUpdateInputsTest(unittest.TestCase):
    def test_update_inputs(self):

        seq_lens_encoder = paddle.to_tensor(
            [[0], [0], [0], [0]], dtype="int32", place=paddle.XPUPlace(0), stop_gradient=True
        )
        seq_lens_decoder = paddle.to_tensor(
            [[27], [29], [31], [31]], dtype="int32", place=paddle.XPUPlace(0), stop_gradient=True
        )
        batch_size = paddle.to_tensor(
            [0, 8191, 16382, 24573], dtype="int32", place=paddle.XPUPlace(0), stop_gradient=True
        )

        a, b = paddle.incubate.nn.functional.blha_get_max_len(seq_lens_encoder, seq_lens_decoder, batch_size)

        print(a)
        print(b)


if __name__ == "__main__":
    unittest.main()
