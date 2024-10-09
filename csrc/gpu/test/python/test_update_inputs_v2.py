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
# limitations under the License.import numpy as np

import unittest

import numpy as np
import paddle
from paddlenlp_ops import set_stop_value_multi_ends_v2, update_inputs_v2, update_inputs

np.random.seed(100)


class UpdateAllOperatorsTest(unittest.TestCase):
    def test_update_all_operations(self):
        # Initialize parameters
        bs = 64
        max_bs = 64
        max_input_length = 6144
        # Initialize tensors for set_stop_value_multi_ends_v2
        topk_ids = paddle.arange(0, bs, dtype="int64")
        next_tokens = paddle.full([bs], 0, dtype="int64")
        step_idx = paddle.full([bs], 0, dtype="int64")

        stop_flags = paddle.to_tensor(np.random.randint(0, 2, [bs]), "bool")
        max_dec_len = paddle.full([bs], 100, dtype="int64")
        step_idx_now = paddle.where(stop_flags, step_idx, step_idx + 1)
        length_cond = paddle.greater_equal(step_idx_now, max_dec_len)
        stop_flags_now = paddle.logical_or(stop_flags, length_cond)

        end_ids = paddle.to_tensor([0, 1, 2, 3, 4, 5], "int64")

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
        is_block_step = np.random.randint(0, 2, [max_bs]).astype("bool")

        # Convert to paddle tensors
        seq_lens_this_time = paddle.to_tensor(seq_lens_this_time)
        seq_lens_encoder = paddle.to_tensor(seq_lens_encoder)
        seq_lens_decoder = paddle.to_tensor(seq_lens_decoder)
        seq_lens_this_time_2 = seq_lens_this_time.clone()
        seq_lens_encoder_2 = seq_lens_encoder.clone()
        seq_lens_decoder_2 = seq_lens_decoder.clone()
        topk_ids_2 = topk_ids.clone()
        next_tokens_2 = next_tokens.clone()
        stop_flags_2 = stop_flags_now.clone()
        end_ids_2 = end_ids.clone()

        # Run set_stop_value_multi_ends_v2
        set_stop_value_multi_ends_v2(topk_ids, stop_flags_now, seq_lens_this_time, end_ids, next_tokens)

        # Initialize tensors for update_inputs
        not_need_stop = paddle.to_tensor(np.array([1], "bool"))
        input_ids = paddle.to_tensor(input_ids_np)
        stop_nums = paddle.to_tensor(stop_nums)
        is_block_step = paddle.to_tensor(is_block_step)

        # Clone tensors for fused_update_inputs
        input_ids_2 = input_ids.clone()
        stop_nums_2 = stop_nums.clone()
        is_block_step_2 = is_block_step.clone()

        # Run update_inputs
        update_inputs(
            stop_flags_now,
            not_need_stop,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            input_ids,
            stop_nums,
            next_tokens,
            is_block_step,
        )

        # Save results from update_inputs
        result2 = {
            "stop_flags": stop_flags_now.numpy(),
            "not_need_stop": not_need_stop.numpy(),
            "seq_lens_this_time": seq_lens_this_time.numpy(),
            "seq_lens_encoder": seq_lens_encoder.numpy(),
            "seq_lens_decoder": seq_lens_decoder.numpy(),
            "input_ids": input_ids.numpy(),
            "stop_nums": stop_nums.numpy(),
            "next_tokens": next_tokens.numpy(),
        }
        print(result2)

        not_need_stop_2 = paddle.to_tensor(np.array([1], "bool"))
        # Run fused_update_inputs
        update_inputs_v2(
            stop_flags_2,
            step_idx,
            not_need_stop_2,
            seq_lens_this_time_2,
            seq_lens_encoder_2,
            seq_lens_decoder_2,
            max_dec_len,
            input_ids_2,
            stop_nums_2,
            topk_ids_2,
            is_block_step_2,
            end_ids_2,
            next_tokens_2,
        )

        # Save results from fused_update_inputs
        result3 = {
            "stop_flags": stop_flags_2.numpy(),
            "not_need_stop": not_need_stop_2.numpy(),
            "seq_lens_this_time": seq_lens_this_time_2.numpy(),
            "seq_lens_encoder": seq_lens_encoder_2.numpy(),
            "seq_lens_decoder": seq_lens_decoder_2.numpy(),
            "input_ids": input_ids_2.numpy(),
            "stop_nums": stop_nums_2.numpy(),
            "next_tokens": next_tokens_2.numpy(),
        }

        # Compare the results between update_inputs and fused_update_inputs
        for key in result2:
            np.testing.assert_array_equal(result2[key], result3[key], err_msg=f"Mismatch in {key}")

        print("All results from `update_inputs` and `fused_update_inputs` are identical.")


if __name__ == "__main__":
    unittest.main()
