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
from collections import defaultdict
from typing import Tuple

import numpy as np
import paddle
from utils.comm_utils import masked_whiten


@paddle.no_grad()
def compute_grpo_advantages(
    rewards: paddle.Tensor,
    index: np.ndarray,
    sequence_mask: paddle.Tensor,
    response_length: int,
    epsilon: float = 1e-6,
):
    """
    计算每个prompt的GRPO优势。

    Args:
        rewards (paddle.Tensor, shape=[batch_size]): 回报，单位为float。
        index (np.ndarray, shape=[batch_size]): 每个样本对应的prompt索引，类型为int。
        sequence_mask (paddle.Tensor, shape=[batch_size, response_length]): 序列掩码，用于标记每个时间步是否有效，类型为bool。
        response_length (int): 每个样本的响应长度。
        epsilon (float, optional, default=1e-6): 避免除以0的值，默认为1e-6。

    Returns:
        rewards (paddle.Tensor, shape=[batch_size, response_length]): GRPO优势，单位为float。

    Raises:
        ValueError (ValueError): 如果没有在给定的prompt索引中有分数。
    """
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    batch_size = rewards.shape[0]

    for i in range(batch_size):
        id2score[index[i]].append(rewards[i])
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = paddle.to_tensor(0.0, dtype=rewards.dtype)
            id2std[idx] = paddle.to_tensor(1.0, dtype=rewards.dtype)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = paddle.mean(paddle.stack(id2score[idx]))
            id2std[idx] = paddle.std(paddle.stack(id2score[idx]))
        else:
            raise ValueError(f"No score in prompt index: {idx}")
    for i in range(batch_size):
        rewards[i] = (rewards[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    rewards = rewards.unsqueeze(-1).tile([1, response_length]) * sequence_mask
    return rewards


@paddle.no_grad()
def compute_reinforce_plus_plus_advantages_and_returns(
    rewards: paddle.Tensor,
    eos_mask: paddle.Tensor,
    gamma: float,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Compute reinforce_plus_plus_advantages_and_returns."""
    length = rewards.shape[-1]
    returns = paddle.zeros_like(rewards)
    running_return = 0
    for t in reversed(range(length)):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        running_return = running_return * eos_mask[:, t]

    advantages = masked_whiten(returns, eos_mask)
    advantages = advantages * eos_mask
    return advantages, returns
