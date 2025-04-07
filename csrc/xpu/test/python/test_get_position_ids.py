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
from paddlenlp_ops import update_inputs

np.random.seed(2023)
class GetUpdateInputsTest(unittest.TestCase):
    def test_update_inputs(self):
        bs = 48
        max_bs = 64
        max_input_length = 6144

        stop_flags = np.random.randint(0, 2, max_bs).astype("bool")
        not_need_stop = np.array([1], "bool")
        seq_lens_this_time = np.zeros([bs], "int32")
        seq_lens_encoder = np.zeros([max_bs], "int32")
        seq_lens_decoder = np.zeros([max_bs], "int32")
        for i in range(bs):
            if i % 2 == 0:
                seq_lens_encoder[i] = i
                seq_lens_this_time[i] = i
            else:
                seq_lens_decoder[i] = i
                seq_lens_this_time[i] = 1
        input_ids_np = np.random.randint(1, 10, [max_bs, max_input_length], "int64")
        stop_nums = np.array([max_bs], "int64")
        next_tokens = np.random.randint(1, 10, [max_bs], "int64")
        is_block_step = np.random.randint(0, 2, [max_bs]).astype("bool")

        stop_flags = paddle.to_tensor(stop_flags)
        not_need_stop = paddle.to_tensor(not_need_stop, place=paddle.CPUPlace())
        seq_lens_this_time = paddle.to_tensor(seq_lens_this_time)
        seq_lens_encoder = paddle.to_tensor(seq_lens_encoder)
        seq_lens_decoder = paddle.to_tensor(seq_lens_decoder)
        input_ids = paddle.to_tensor(input_ids_np)
        stop_nums = paddle.to_tensor(stop_nums)
        next_tokens = paddle.to_tensor(next_tokens)
        is_block_step = paddle.to_tensor(is_block_step)

        print("stop_flags:\n", stop_flags)
        print("not_need_stop:\n", not_need_stop)
        print("seq_lens_this_time:\n", seq_lens_this_time)
        print("seq_lens_encoder:\n", seq_lens_encoder)
        print("seq_lens_decoder:\n", seq_lens_decoder)
        print("input_ids:\n", input_ids)
        print("stop_nums:\n", stop_nums)
        print("next_tokens:\n", next_tokens)
        print("is_block_step:\n", is_block_step)

        update_inputs(
            stop_flags,
            not_need_stop,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            input_ids,
            stop_nums,
            next_tokens,
            is_block_step
        )

        print("-" * 50)
        print("stop_flags:\n", stop_flags)
        print("not_need_stop:\n", not_need_stop)
        print("seq_lens_this_time:\n", seq_lens_this_time)
        print("seq_lens_encoder:\n", seq_lens_encoder)
        print("seq_lens_decoder:\n", seq_lens_decoder)
        print("input_ids:\n", input_ids)
        print("stop_nums:\n", stop_nums)
        print("next_tokens:\n", next_tokens)

        ref_not_need_stop_out = np.array([True])
        ref_seq_lens_this_time_out = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
                                            1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1], "int32")
        ref_seq_lens_encoder_out = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "int32")
        ref_seq_lens_decoder_out = np.array([0, 0, 2, 0, 0, 6, 0, 8, 8, 10, 0, 12, 12, 0, 0, 0, 0, 0, 0, 0, 20, 22, 0, 24,
                                            24, 0, 26, 28, 0, 0, 0, 32, 32, 0, 34, 0, 0, 38, 0, 40, 0, 0, 42, 0, 0, 46, 46, 48,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "int32")
        input_ids_np[:, 0] = np.array([6, 5, 9, 8, 6, 2, 8, 1, 3, 1, 3, 6, 9, 8, 1, 9, 1, 8, 8, 6, 7, 6, 5, 3,
                                    5, 9, 3, 6, 3, 9, 8, 8, 8, 8, 4, 8, 7, 4, 2, 3, 5, 8, 4, 2, 5, 6, 8, 9,
                                    6, 7, 4, 2, 4, 6, 2, 3, 4, 9, 7, 2, 1, 8, 7, 8], "int64")

        assert not_need_stop.numpy() == ref_not_need_stop_out, 'Check not_need_stop failed.'
        assert np.all(seq_lens_this_time.numpy()
                    == ref_seq_lens_this_time_out), 'Check seq_lens_this_time failed.'
        assert np.all(seq_lens_encoder.numpy()
                    == ref_seq_lens_encoder_out), 'Check seq_lens_encoder failed.'
        assert np.all(seq_lens_decoder.numpy()
                    == ref_seq_lens_decoder_out), 'Check seq_lens_decoder failed.'
        assert np.all(input_ids.numpy()
                    == input_ids_np), 'Check input_ids failed.'

if __name__ == '__main__':
    unittest.main()
