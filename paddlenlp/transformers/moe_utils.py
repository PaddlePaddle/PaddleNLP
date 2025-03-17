# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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


def topk_to_permuted_indices(x, num_tokens_per_expert_list, topk):
    x = paddle.flatten(x)
    prob_permuted_indices = paddle.concat(
        [
            paddle.tensor.search._restrict_nonzero(x == i, total_true_num)
            for i, total_true_num in enumerate(num_tokens_per_expert_list)
        ]
    ).flatten()
    token_permuted_indices = prob_permuted_indices // topk
    return token_permuted_indices, prob_permuted_indices


def permute(
    tokens,
    token_permuted_indices,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (paddle.Tensor): The input token tensor, [num_tokens, hidden].
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    permuted_input = paddle.gather(tokens, token_permuted_indices)
    return permuted_input


def unpermute(
    permuted_tokens: paddle.Tensor,
    token_permuted_indices: paddle.Tensor,
    prob_permuted_indices: paddle.Tensor,
    restore_shape: paddle.shape,
    probs: paddle.Tensor = None,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (paddle.Tensor): The permuted token tensor.
        token_permuted_indices (paddle.Tensor): The indices used to sort the tokens.
        restore_shape (paddle.shape): The shape of the unpermuted tensor.
        probs (paddle.Tensor, optional): The unpermuted probs tensor,
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        paddle.Tensor: The tokens restored to their original order.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    _, hidden = restore_shape

    if probs is not None:
        permuted_probs = paddle.gather(probs.flatten(), prob_permuted_indices)
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = paddle.zeros(restore_shape, dtype=permuted_tokens.dtype)
    # Scatter add the permuted_input back to the original positions
    output_tokens.put_along_axis_(
        axis=0,
        indices=token_permuted_indices.unsqueeze(1).expand([-1, hidden]),
        values=permuted_tokens,
        reduce="add",
        include_self=True,
    )
    return output_tokens
