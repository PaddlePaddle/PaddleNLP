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

from typing import Optional

import paddle

try:
    from paddle import scatter_add_
except ImportError:
    scatter_add_ = None


def permute(
    tokens,
    routing_map,
    num_out_tokens: Optional[int] = None,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (paddle.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (paddle.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.cast(paddle.bool).T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = paddle.arange(num_tokens).unsqueeze(0).expand([num_experts, -1])
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(axis=0, index=sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    permuted_tokens: paddle.Tensor,
    sorted_indices: paddle.Tensor,
    restore_shape: paddle.shape,
    probs: paddle.Tensor = None,
    routing_map: paddle.Tensor = None,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (paddle.Tensor): The permuted token tensor.
        sorted_indices (paddle.Tensor): The indices used to sort the tokens.
        restore_shape (paddle.shape): The shape of the unpermuted tensor.
        probs (paddle.Tensor, optional): The unpermuted probs tensor,
        routing_map (paddle.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        paddle.Tensor: The tokens restored to their original order.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    _, hidden = restore_shape

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = paddle.zeros(restore_shape, dtype=permuted_tokens.dtype)
    # Scatter add the permuted_input back to the original positions
    if scatter_add_ is not None:
        scatter_add_(output_tokens, sorted_indices, permuted_tokens)
    else:
        output_tokens.scatter_(index=sorted_indices, updates=permuted_tokens, overwrite=False)
    return output_tokens


def topk_to_permuted_indices_single(x, num_tokens, expert_id, topk):
    """
    Convert the topk indices to permuted indices.
    """
    x = paddle.flatten(x)
    prob_permuted_indices = paddle.tensor.search._restrict_nonzero(x == expert_id, num_tokens).flatten()
    token_permuted_indices = prob_permuted_indices // topk
    return token_permuted_indices, prob_permuted_indices


def topk_to_permuted_indices(x, num_tokens_per_expert_list, topk):
    """
    Convert the topk indices to permuted indices.
    """
    x = paddle.flatten(x)
    prob_permuted_indices = paddle.concat(
        [
            paddle.tensor.search._restrict_nonzero(x == i, total_true_num)
            for i, total_true_num in enumerate(num_tokens_per_expert_list)
        ]
    ).flatten()
    token_permuted_indices = prob_permuted_indices // topk
    return token_permuted_indices, prob_permuted_indices


class FakeClone(paddle.autograd.PyLayer):
    """
    manual_backward中, 为了保留局部的计算图做临时反向计算
    需要把manual_backward的output给clone出来, 这个clone
    本质上不需要output的值, 而是需要拿到output身上的计算图

    但调用paddle.clone会做一次额外的数据拷贝, 这是没必要的
    FakeClone可以免去这个数据拷贝, 实现摘取计算图的目的
    """

    @staticmethod
    def forward(ctx, input):
        """forward"""
        if input.is_contiguous():
            fake_output = paddle.empty_like(input)
            input._share_buffer_to(fake_output)
        else:
            fake_output = input.clone()
        return fake_output

    @staticmethod
    def backward(ctx, grad_output):
        """backward"""
        return grad_output


class FakeGather(paddle.autograd.PyLayer):
    """
    临时绕开gather 0size索引的coredump问题
    """

    @staticmethod
    def forward(ctx, input, indices):
        """forward"""
        assert len(indices.shape) == 1
        ctx.save_for_backward(indices)
        ctx.input_shape = input.shape
        if indices.shape[0] == 0:
            out_shape = input.shape
            out_shape[0] = 0
            return paddle.zeros(out_shape, dtype=input.dtype)
        return paddle.index_select(input, axis=0, index=indices)

    @staticmethod
    def backward(ctx, grad_output):
        """backward"""
        indices = ctx.saved_tensor()
        input_shape = ctx.input_shape
        grad_input = paddle.zeros(input_shape, dtype=grad_output.dtype)
        if indices.shape[0] != 0:
            if scatter_add_ is not None:
                scatter_add_(grad_input, indices.unsqueeze(-1), grad_output)
            else:
                paddle.scatter_(grad_input, indices.unsqueeze(-1), grad_output, overwrite=False)
        return grad_input, None
