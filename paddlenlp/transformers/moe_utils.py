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

from .fp8_utils import dequantize_fp8_to_fp32


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


class PermuteNode:
    def __init__(self, token_dispatcher, name="permute"):
        self.token_dispatcher = token_dispatcher
        self.name = name

    def forward(self, hidden_states, hidden_states_scale, dispatched_indices):
        self.token_dispatcher._comm_manager.hidden_shape_before_permute = hidden_states.shape
        self.hidden_shape_before_permute = hidden_states.shape
        self.token_permuted_indices, self.prob_permuted_indices = topk_to_permuted_indices(
            dispatched_indices,
            self.token_dispatcher._comm_manager.tokens_per_expert,
            self.token_dispatcher._comm_manager.router_topk,
        )
        hidden_states = permute(hidden_states, self.token_permuted_indices)
        # permute scale
        hidden_states_scale = permute(hidden_states_scale, self.token_permuted_indices)

        return hidden_states, hidden_states_scale, self.token_permuted_indices, self.prob_permuted_indices

    def backward(self, out_grad, dispatched_probs):
        input_dtype = out_grad.dtype
        hidden_states_grad = unpermute(
            permuted_tokens=out_grad,
            token_permuted_indices=self.token_permuted_indices,
            prob_permuted_indices=self.prob_permuted_indices,
            restore_shape=self.hidden_shape_before_permute,
            probs=dispatched_probs,
        )
        return hidden_states_grad.to(input_dtype)


class UnPermuteNode:
    def __init__(self, token_dispatcher, name="unpermute"):
        self.token_dispatcher = token_dispatcher
        self.name = name

    def forward(
        self,
        hidden_states,
        token_permuted_indices,
        prob_permuted_indices,
        dispatched_probs,
    ):
        self.token_permuted_indices = token_permuted_indices
        self.input_dtype = hidden_states.dtype
        self.hidden_states = hidden_states
        self.prob_permuted_indices = prob_permuted_indices
        self.dispatched_probs = dispatched_probs
        # permute
        _, self.hidden = self.token_dispatcher._comm_manager.hidden_shape_before_permute

        self.faltten_dispatched_probs = self.dispatched_probs.flatten()

        self.permuted_probs = paddle.gather(self.faltten_dispatched_probs, self.prob_permuted_indices)
        self.permuted_tokens = self.hidden_states * self.permuted_probs.unsqueeze(-1)
        self.permuted_tokens_dtype = self.permuted_tokens.dtype

        # Create an output tensor filled with zeros
        output_tokens = paddle.zeros(
            self.token_dispatcher._comm_manager.hidden_shape_before_permute, dtype=self.permuted_tokens_dtype
        )
        # Scatter add the permuted_input back to the original positions
        output_tokens.put_along_axis_(
            axis=0,
            indices=self.token_permuted_indices.unsqueeze(1).expand([-1, self.hidden]),
            values=self.permuted_tokens,
            reduce="add",
            include_self=True,
        )
        with paddle.base.device_guard("cpu"):
            self.output_tokens = paddle.empty(shape=output_tokens.shape, dtype=output_tokens.dtype)

        return output_tokens.to(self.input_dtype)

    def backward(self, out_grad, out_grad_scale):
        hidden_states_grad = paddle.gather(out_grad, self.token_permuted_indices)

        output_tokens_grad = dequantize_fp8_to_fp32(out_grad, out_grad_scale)

        _, permuted_tokens_grad = paddle._C_ops.put_along_axis_grad(
            self.output_tokens,
            self.token_permuted_indices.unsqueeze(1).expand([-1, self.hidden]),
            self.permuted_tokens,
            self.output_tokens,
            output_tokens_grad,
            0,
            "add",
            True,
        )

        permuted_probs_grad = (permuted_tokens_grad * self.hidden_states).sum(axis=-1)

        faltten_dispatched_probs_grad = paddle._C_ops.gather_grad(
            self.faltten_dispatched_probs, self.prob_permuted_indices, permuted_probs_grad, 0
        )

        dispatched_probs_grad = paddle._C_ops.flatten_grad(self.dispatched_probs, faltten_dispatched_probs_grad)

        return hidden_states_grad, dispatched_probs_grad
