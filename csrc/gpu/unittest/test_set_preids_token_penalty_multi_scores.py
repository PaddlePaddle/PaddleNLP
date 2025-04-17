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
from paddlenlp_ops import (
    get_token_penalty_multi_scores_v2,
    set_preids_token_penalty_multi_scores,
    set_value_by_flags_and_idx_v2,
)

paddle.seed(2023)


class SetPreidsTokenPenaltyMultiScores(unittest.TestCase):
    def test_set_preids_token_penalty_multi_scores_operations(self):
        pre_ids = paddle.to_tensor([[1, 9, 3, 4, 5, 6, 7, -1, -1, -1], [1, 9, 7, 6, 5, 4, -1, -1, -1, -1]], "int64")
        pre_ids_2 = pre_ids.clone()
        input_ids = paddle.to_tensor(
            [[1, 9, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1], [1, 9, 7, 6, 5, 4, -1, -1, -1, -1, -1, -1, -1]], "int64"
        )
        input_ids_2 = input_ids.clone()
        seq_lens_this_time = paddle.to_tensor([1, 1], "int32")
        seq_lens_encoder = paddle.to_tensor([1, 1], "int32")
        seq_lens_encoder_2 = seq_lens_encoder.clone()
        seq_lens_decoder = paddle.to_tensor([1, 1], "int32")
        seq_lens_decoder_2 = seq_lens_decoder.clone()
        step_idx = paddle.to_tensor([1, 1], "int64")
        step_idx_2 = step_idx.clone()
        stop_flags = paddle.to_tensor([0, 1], "bool")
        stop_flags_2 = stop_flags.clone()
        logits = paddle.to_tensor(
            [[0.1, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.1, 0.1], [0.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.1, 0.1, 0.1, 0.1]],
            "float32",
        )
        logits_2 = logits.clone()
        penalty_scores = paddle.to_tensor([1.0, 1.0], "float32")
        penalty_scores_2 = penalty_scores.clone()
        frequency_scores = paddle.to_tensor([0.1, 0.1], "float32")
        frequency_scores_2 = frequency_scores.clone()
        presence_scores = paddle.to_tensor([0.0, 0.0], "float32")
        presence_scores_2 = presence_scores.clone()
        temperatures = paddle.to_tensor([0.5, 0.25], "float32")
        temperatures_2 = temperatures.clone()
        bad_tokens = paddle.to_tensor([0, 1], "int64")
        bad_tokens_2 = bad_tokens.clone()
        cur_len = paddle.to_tensor([7, 6], "int64")
        cur_len_2 = cur_len.clone()
        min_len = paddle.to_tensor([1, 8], "int64")
        min_len_2 = min_len.clone()
        eos_token_id = paddle.to_tensor([2, 9], "int64")
        eos_token_id_2 = eos_token_id.clone()

        # Run set_value_by_flags_and_idx_v2 and get_token_penalty_multi_scores_v2
        print("Running set_value_by_flags_and_idx_v2...")
        set_value_by_flags_and_idx_v2(
            pre_ids, input_ids, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx, stop_flags
        )

        get_token_penalty_multi_scores_v2(
            pre_ids,
            logits,
            penalty_scores,
            frequency_scores,
            presence_scores,
            temperatures,
            bad_tokens,
            cur_len,
            min_len,
            eos_token_id,
        )

        print("Results after first operation:")
        print("pre_ids\n", pre_ids.numpy())
        print("logits\n", logits.numpy())
        print("penalty_scores\n", penalty_scores.numpy())
        print("frequency_scores\n", frequency_scores.numpy())
        print("presence_scores\n", presence_scores.numpy())
        print("temperatures\n", temperatures.numpy())
        print("bad_tokens\n", bad_tokens.numpy())
        print("cur_len\n", cur_len.numpy())
        print("min_len\n", min_len.numpy())
        print("eos_token_id\n", eos_token_id.numpy())

        first_pre_ids = pre_ids.numpy()
        first_logits = logits.numpy()

        # Run set_preids_token_penalty_multi_scores
        print("Running set_preids_token_penalty_multi_scores...")
        set_preids_token_penalty_multi_scores(
            pre_ids_2,
            input_ids_2,
            seq_lens_encoder_2,
            seq_lens_decoder_2,
            step_idx_2,
            stop_flags_2,
            logits_2,
            penalty_scores_2,
            frequency_scores_2,
            presence_scores_2,
            temperatures_2,
            bad_tokens_2,
            cur_len_2,
            min_len_2,
            eos_token_id_2,
        )

        print("Results after second operation:")
        print("pre_ids_2\n", pre_ids_2.numpy())
        print("logits_2\n", logits_2.numpy())
        print("penalty_scores_2\n", penalty_scores_2.numpy())
        print("frequency_scores_2\n", frequency_scores_2.numpy())
        print("presence_scores_2\n", presence_scores_2.numpy())
        print("temperatures_2\n", temperatures_2.numpy())
        print("bad_tokens_2\n", bad_tokens_2.numpy())
        print("cur_len_2\n", cur_len_2.numpy())
        print("min_len_2\n", min_len_2.numpy())
        print("eos_token_id_2\n", eos_token_id_2.numpy())

        second_pre_ids = pre_ids_2.numpy()
        second_logits = logits_2.numpy()

        np.testing.assert_array_equal(
            first_pre_ids, second_pre_ids, err_msg="pre_ids arrays are different between runs"
        )
        np.testing.assert_array_equal(first_logits, second_logits, err_msg="logits arrays are different between runs")
        print("Test passed: The two runs produced identical results.")


if __name__ == "__main__":
    unittest.main()
