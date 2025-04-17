# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def apply_overlong_penalty(response_length, max_dec_len, overlong_buffer_len, penalty_factor):
    """
    Apply length penalty to overlong responses.

    Args:
        response_length (paddle.Tensor): Tensor of shape (B,) indicating the length of each response.
        max_dec_len (int): The maximum allowed decoding length.
        overlong_buffer_len (int): The allowed buffer before applying penalty.
        penalty_factor (float): The penalty factor to scale the length overflow.

    Returns:
        paddle.Tensor: A tensor of shape (B,) representing the length penalty for each response.
    """
    expected_len = max_dec_len - overlong_buffer_len
    exceed_len = response_length - expected_len

    reward_penalty = -exceed_len / overlong_buffer_len * penalty_factor
    # Only apply negative penalty if response exceeds limit, otherwise zero
    overlong_penalty = paddle.minimum(reward_penalty, paddle.zeros_like(reward_penalty))

    return overlong_penalty
