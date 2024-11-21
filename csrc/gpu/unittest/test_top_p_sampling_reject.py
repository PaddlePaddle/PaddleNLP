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
from paddlenlp_ops import top_p_sampling_reject

paddle.seed(2023)

batch_size = 3
vocab_size = 40080
max_rounds = 32

class SetPreidsTokenPenaltyMultiScores(unittest.TestCase):
    def test_top_p_sampling_reject_case1(self):
        # top_p为1, 不同seed
        pre_norm_prob_np = np.random.rand(batch_size, vocab_size).astype(np.float32)

        paddle_pre_norm_prob = paddle.to_tensor(pre_norm_prob_np)
        paddle_norm_prob = paddle_pre_norm_prob / paddle_pre_norm_prob.sum(axis=-1, keepdim=True)
        top_p_paddle = paddle.full((batch_size,), 1)
        samples = top_p_sampling_reject(paddle_norm_prob, top_p_paddle, 0)
        print(samples)
        samples = top_p_sampling_reject(paddle_norm_prob, top_p_paddle, 1024)
        print(samples)
        samples = top_p_sampling_reject(paddle_norm_prob, top_p_paddle, 2033)
        print(samples)

    def test_top_p_sampling_reject_case2(self):
        # top_p为0
        pre_norm_prob_np = np.random.rand(batch_size, vocab_size).astype(np.float32)

        paddle_pre_norm_prob = paddle.to_tensor(pre_norm_prob_np)
        paddle_norm_prob = paddle_pre_norm_prob / paddle_pre_norm_prob.sum(axis=-1, keepdim=True)
        top_p_paddle = paddle.full((batch_size,), 0)
        samples = top_p_sampling_reject(paddle_norm_prob, top_p_paddle, 0)
        print(samples)

    def test_top_p_sampling_reject_case3(self):
        # 不同batch的top_p值不同
        pre_norm_prob_np = np.random.rand(batch_size, vocab_size).astype(np.float32)

        paddle_pre_norm_prob = paddle.to_tensor(pre_norm_prob_np)
        paddle_norm_prob = paddle_pre_norm_prob / paddle_pre_norm_prob.sum(axis=-1, keepdim=True)
        top_p_paddle = paddle.uniform(shape=[batch_size,1], min=0, max=1)
        samples = top_p_sampling_reject(paddle_norm_prob, top_p_paddle, 0)
        print(samples)

if __name__ == "__main__":
    unittest.main()