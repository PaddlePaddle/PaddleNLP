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

import numpy as np
import paddle
import unittest
from paddlenlp_ops import set_value_by_flags_and_idx_v2

paddle.seed(2023)

class GetStopValueMultiEndsV2Test(unittest.TestCase):
    def test_set_stop_value_multi_ends_v2(self):
        pre_ids_all = paddle.to_tensor([[1, 9, 3, 4, 5, 6, 7, -1, -1, -1], [1, 9, 7, 6, 5, 4, -1, -1, -1, -1]], "int64")
        input_ids = paddle.to_tensor([[1, 9, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1], [1, 9, 7, 6, 5, 4, -1, -1, -1, -1, -1, -1, -1]], "int64")
        seq_lens_this_time = paddle.to_tensor([1, 1], "int32")
        seq_lens_encoder = paddle.to_tensor([1, 1], "int32")
        seq_lens_decoder = paddle.to_tensor([1, 1], "int32")
        step_idx = paddle.to_tensor([1, 1], "int64")
        stop_flags = paddle.to_tensor([0, 1], "bool")
        print("pre_ids_all\n", pre_ids_all)
        set_value_by_flags_and_idx_v2(pre_ids_all, input_ids, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx, stop_flags)
        print("pre_ids_all\n", pre_ids_all)
        print("input_ids\n", input_ids)
        print("seq_lens_this_time\n", seq_lens_this_time)
        print("seq_lens_encoder\n", seq_lens_encoder)
        print("seq_lens_decoder\n", seq_lens_decoder)
        print("step_idx\n", step_idx)
        print("stop_flags\n", stop_flags)

        ref_pre_ids_all = np.array(
            [
                [
                    1,
                    1,
                    3,
                    4,
                    5,
                    6,
                    7,
                    -1,
                    -1,
                    -1,
                ],
                [
                    1,
                    9,
                    7,
                    6,
                    5,
                    4,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
            ],
            "int64",
        )
        diff_pre_ids_all = np.sum(np.abs(ref_pre_ids_all - pre_ids_all.numpy()))
        print("diff_pre_ids_all\n", diff_pre_ids_all)
        assert diff_pre_ids_all == 0, 'Check failed.'

if __name__ == '__main__':
    unittest.main()
