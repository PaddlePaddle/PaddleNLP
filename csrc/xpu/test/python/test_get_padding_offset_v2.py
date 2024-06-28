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
from paddlenlp_ops import get_padding_offset_v2

np.random.seed(2023)

class GetPaddingOffsetV2Test(unittest.TestCase):
    def test_get_padding_offset_v2(self):
        max_len = 10
        seq_lens = np.array([4, 3, 6], "int32").reshape(-1, 1)
        cum_offset = np.cumsum((max_len - seq_lens).flatten(), -1, "int32")
        token_num = np.sum(seq_lens)
        bs = seq_lens.shape[0]
        input_ids = np.zeros([bs, max_len], "int64")
        for i in range(bs):
            ids_len = seq_lens[i, 0]
            input_ids[i, 0:ids_len] = np.random.randint(1, 10, seq_lens[i, 0], "int64")

        x_remove_padding, cum_offsets_out, padding_offset, cu_seqlens_q, cu_seqlens_k = get_padding_offset_v2(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(cum_offset),
            paddle.to_tensor(token_num),
            paddle.to_tensor(seq_lens),
        )

        print("input_ids:\n", input_ids)
        print("cum_offset:\n", cum_offset)
        print("token_num:\n", token_num)
        print("seq_lens:\n", seq_lens)
        print("x_remove_padding:\n", x_remove_padding)
        print("cum_offsets_out:\n", cum_offsets_out)
        print("padding_offset:\n", padding_offset)
        print("cu_seqlens_q:\n", cu_seqlens_q)
        print("cu_seqlens_k:\n", cu_seqlens_k)

        ref_x_remove_padding = np.array(
            [8, 7, 8, 2, 4, 5, 5, 7, 6, 1, 7, 2, 6], "int64")
        ref_cum_offsets_out = np.array([0, 6, 13], "int32")
        ref_padding_offset = np.array(
            [0, 0, 0, 0, 6, 6, 6, 13, 13, 13, 13, 13, 13], "int32")
        ref_cu_seqlens_q = np.array([0 , 4 , 7 , 13], "int32")
        ref_cu_seqlens_k = np.array([0 , 4 , 7 , 13], "int32")

        assert sum(ref_x_remove_padding
                - x_remove_padding) == 0, 'Check x_remove_padding failed.'
        assert sum(ref_cum_offsets_out
                - cum_offsets_out) == 0, 'Check cum_offsets_out failed.'
        assert sum(ref_padding_offset
                - padding_offset) == 0, 'Check padding_offset failed.'
        assert sum(ref_cu_seqlens_q - cu_seqlens_q) == 0, 'Check cu_seqlens_q failed.'
        assert sum(ref_cu_seqlens_k - cu_seqlens_k) == 0, 'Check cu_seqlens_k failed.'

if __name__ == '__main__':
    unittest.main()