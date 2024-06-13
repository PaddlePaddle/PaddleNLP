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
from paddlenlp_ops import set_stop_value_multi_ends_v2
np.random.seed(1)

class GetStopValueMultiEndsV2Test(unittest.TestCase):
    def test_set_stop_value_multi_ends_v2(self):
        bs = 64

        # test beam_search=False
        topk_ids = paddle.arange(0, bs, dtype="int64")
        next_tokens = paddle.full([bs], 0, dtype="int64")
        stop_flags = paddle.to_tensor(np.random.randint(0, 2, [bs]), "bool")
        seq_lens = paddle.to_tensor(np.random.randint(0, 5, [bs]), "int32")
        end_ids = paddle.to_tensor([0, 1, 2, 3, 4, 5], "int64")
        print("topk_ids\n", topk_ids)
        print("next_tokens\n", next_tokens)
        print("stop_flags\n", stop_flags)
        set_stop_value_multi_ends_v2(topk_ids, stop_flags,
                                seq_lens, end_ids, next_tokens)
        print("topk_ids\n", topk_ids)
        print("next_tokens\n", next_tokens)
        print("stop_flags\n", stop_flags)
        print("seq_lens\n", seq_lens)
        print("end_ids\n", end_ids)

        ref_topk_ids = np.array(
            [0, 0, 2, 3, -1, 0, 0, 0, 0, 9, 10, 0, 12, 0, -1, 15, 16, 0,
            18, 19, 20, 0, 22, 23, 0, 25, 26, 27, -1, 29, 30, 31, 0, 0, 0, -1,
            -1, 37, 38, 39, -1, -1, 0, 0, 0, 0, 46, -1, 0, 49, 50, 0, 52, 53,
            0, -1, 0, 57, -1, 59, 60, 0, 0, 63],
            "int64",
        )
        ref_next_tokens = np.array(
            [0, 0, 2, 3, 0, 0, 0, 0, 0, 9, 10, 0, 12, 0, 0, 15, 16, 0,
            18, 19, 20, 0, 22, 23, 0, 25, 26, 27, 0, 29, 30, 31, 0, 0, 0, 0,
            0, 37, 38, 39, 0, 0, 0, 0, 0, 0, 46, 0, 0, 49, 50, 0, 52, 53,
            0, 0, 0, 57, 0, 59, 60, 0, 0, 63],
            "int64",
        )
        ref_stop_flags = np.array(
            [True, True, True, True, True, True, True, True, True, False,
            False, True, False, True, True, False, False, True, False, False,
            False, True, False, False, True, False, False, False, True, False,
            False, False, True, True, True, True, True, False, False, False,
            True, True, True, True, True, True, False, True, True, False,
            False, True, False, False, True, True, True, False, True, False,
            False, True, True, False],
            "bool",
        )
        diff_topk_ids = np.sum(np.abs(ref_topk_ids - topk_ids.numpy()))
        print("diff_topk_ids\n", diff_topk_ids)
        assert diff_topk_ids == 0, 'Check failed.'
        diff_next_tokens = np.sum(np.abs(ref_next_tokens - next_tokens.numpy()))
        print("diff_next_tokens\n", diff_next_tokens)
        assert diff_next_tokens == 0, 'Check failed.'
        diff_stop_flags = np.sum(np.abs(ref_stop_flags.astype(
            np.int32) - stop_flags.numpy().astype(np.int32)))
        print("diff_stop_flags\n", diff_stop_flags)
        assert diff_stop_flags == 0, 'Check failed.'

        # test beam_search=True
        # topk_ids = paddle.arange(0, bs, dtype="int64")
        # next_tokens = paddle.full([bs], 0, dtype="int64")
        # stop_flags = paddle.to_tensor(np.random.randint(0, 2, [bs]), "bool")
        # seq_lens = paddle.to_tensor(np.random.randint(0, 5, [bs]), "int32")
        # end_ids = paddle.to_tensor([0, 1, 2, 3, 4, 5], "int64")
        # print("topk_ids\n", topk_ids)
        # print("next_tokens\n", next_tokens)
        # print("stop_flags\n", stop_flags)
        # set_stop_value_multi_ends_v2(topk_ids, stop_flags,
        #                           seq_lens, end_ids, next_tokens, True)
        # print("topk_ids\n", topk_ids)
        # print("next_tokens\n", next_tokens)
        # print("stop_flags\n", stop_flags)
        # print("seq_lens\n", seq_lens)
        # print("end_ids\n", end_ids)

        # ref_topk_ids = np.array(
        #     [0, 1, 2, 3, 4, 0, 6, 7, -1, 9, 10, 0, -1, 13, 14, 15, 0, 17,
        #      18, 19, 20, 0, 22, 23, 24, 25, -1, -1, 28, 29, 0, 0, -1, 33, 34, 35,
        #      36, 37, 0, -1, 0, 41, -1, 0, 44, 45, 46, 0, 0, 49, 0, 0, 0, 53,
        #      0, 0, 0, 0, 58, -1, 60, 61, -1, 63],
        #     "int64",
        # )
        # ref_next_tokens = np.array(
        #     [0, 1, 2, 3, 4, 0, 6, 7, 0, 9, 10, 0, 0, 13, 14, 15, 0, 17,
        #      18, 19, 20, 0, 22, 23, 24, 25, 0, 0, 28, 29, 0, 0, 0, 33, 34, 35,
        #      36, 37, 0, 0, 0, 41, 0, 0, 44, 45, 46, 0, 0, 49, 0, 0, 0, 53,
        #      0, 0, 0, 0, 58, 0, 60, 61, 0, 63],
        #     "int64",
        # )
        # ref_stop_flags = np.array(
        #     [False, False, False, False, False, True, False, False, True, False,
        #      False, True, True, False, False, False, True, False, False, False,
        #      False, True, False, False, False, False, True, True, False, False,
        #      True, True, True, False, False, False, False, False, True, True,
        #      True, False, True, True, False, False, False, True, True, False,
        #      True, True, True, False, True, True, True, True, False, True,
        #      False, False, True, False],
        #     "bool",
        # )
        # diff_topk_ids = np.sum(np.abs(ref_topk_ids - topk_ids.numpy()))
        # print("diff_topk_ids\n", diff_topk_ids)
        # assert diff_topk_ids == 0, 'Check failed.'
        # diff_next_tokens = np.sum(np.abs(ref_next_tokens - next_tokens.numpy()))
        # print("diff_next_tokens\n", diff_next_tokens)
        # assert diff_next_tokens == 0, 'Check failed.'
        # diff_stop_flags = np.sum(np.abs(ref_stop_flags.astype(
        #     np.int32) - stop_flags.numpy().astype(np.int32)))
        # print("diff_stop_flags\n", diff_stop_flags)
        # assert diff_stop_flags == 0, 'Check failed.'.'

if __name__ == '__main__':
    unittest.main()