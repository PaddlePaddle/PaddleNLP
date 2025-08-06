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

from ..utils.comm_utils import masked_whiten


@paddle.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: paddle.Tensor,
    values: paddle.Tensor,
    sequence_mask: paddle.Tensor,
    gamma: paddle.Tensor,
    lam: paddle.Tensor,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
    # Modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py

    lastgaelam = 0.0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]

    values = values * sequence_mask
    token_level_rewards = token_level_rewards * sequence_mask

    for t in reversed(range(0, gen_len)):
        next_values = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * next_values - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = paddle.stack(advantages_reversed[::-1], axis=1)

    returns = advantages + values.contiguous()

    return advantages.detach(), returns


@paddle.no_grad()
def compute_grpo_advantages(
    rewards: paddle.Tensor,
    index: np.ndarray,
    sequence_mask: paddle.Tensor,
    response_length: int,
    epsilon: float = 1e-6,
):
    """
    Computes the GRPO advantage for each prompt.

    Args:
        rewards (paddle.Tensor): Rewards tensor with shape [batch_size].
        index (np.ndarray): Array of prompt indices with shape [batch_size].
        sequence_mask (paddle.Tensor): Sequence mask tensor with shape [batch_size, response_length].
        response_length (int): Length of each response.
        epsilon (float): Small value to avoid division by zero, default is 1e-6.

    Returns:
        paddle.Tensor: GRPO advantage tensor with shape [batch_size, response_length].

    Raises:
        ValueError: If there are no scores for a given prompt index.
    """
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    batch_size = rewards.shape[0]

    # Populate the scores for each prompt index.
    for i in range(batch_size):
        id2score[index[i]].append(rewards[i])

    # Compute mean and standard deviation for each prompt index.
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = paddle.to_tensor(0.0, dtype=rewards.dtype)
            id2std[idx] = paddle.to_tensor(1.0, dtype=rewards.dtype)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = paddle.mean(paddle.stack(id2score[idx]))
            id2std[idx] = paddle.std(paddle.stack(id2score[idx]))
        else:
            raise ValueError(f"No score in prompt index: {idx}")

    # Compute the GRPO advantage for each sample.
    for i in range(batch_size):
        rewards[i] = (rewards[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    # Reshape and apply the sequence mask.
    rewards = rewards.unsqueeze(-1).tile([1, response_length]) * sequence_mask
    return rewards


@paddle.no_grad()
def compute_reinforce_plus_plus_advantages_and_returns(
    rewards: paddle.Tensor,  # Rewards tensor.
    eos_mask: paddle.Tensor,  # End-of-sequence mask tensor.
    gamma: float,  # Discount factor.
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Computes the Reinforce++ advantages and returns.

    Args:
        rewards (paddle.Tensor): Rewards tensor.
        eos_mask (paddle.Tensor): End-of-sequence mask tensor.
        gamma (float): Discount factor.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor]: Reinforce++ advantages and returns tensors.
    """
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


@paddle.no_grad()
def add_kl_divergence_regularization(
    prompt: paddle.Tensor,  # size = (B, S) # pylint: disable=unused-argument
    log_probs: paddle.Tensor,  # size = (B, L)
    ref_log_probs: paddle.Tensor,  # size = (B, L)
    reward_score: paddle.Tensor,  # size = (B,)
    sequence_mask: paddle.Tensor,  # size = (B, L)
    kl_coeff: float,
    clip_range_score: float,
) -> paddle.Tensor:
    """
    Calculate the KL divergence regularization gain and add it to the reward.

    Args:
        prompt (paddle.Tensor, shape=(B, S)): The prompt of the input sequence, not used.
        log_probs (paddle.Tensor, shape=(B, L)): The log probability distribution of the current predictions.
        ref_log_probs (paddle.Tensor, shape=(B, L)): The log probability distribution of the baseline predictions.
        reward_score (paddle.Tensor, shape=(B,)): The base reward score based on the prompt and output sequence.
        sequence_mask (paddle.Tensor, shape=(B, L)): The mask of the sequence, used to determine the length of the sequence.

    Returns:
        paddle.Tensor, shape=(B, L): A vector containing the KL divergence regularization gain.
    """

    kl_divergence_estimate = -kl_coeff * (log_probs - ref_log_probs)  # size = (B, L)
    rewards = kl_divergence_estimate  # size = (B, L)
    reward_clip = paddle.clip(  # size = (B,)
        reward_score,
        min=-clip_range_score,
        max=clip_range_score,
    )
    # TODO(guosheng): use scatter_add/put_along_axis
    index = paddle.cumsum(sequence_mask.cast(paddle.int64), axis=-1).argmax(-1, keepdim=True)

    rewards = paddle.put_along_axis(
        rewards,
        index,
        reward_clip.unsqueeze(axis=-1),
        axis=-1,
        reduce="add",
    )
    return rewards, kl_divergence_estimate
