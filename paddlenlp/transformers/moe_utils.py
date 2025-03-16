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
import TokenDispatherUtils as TDU


class IndicesToMultihot(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, indices, probs, router_topk, num_local_experts):
        ctx.save_for_backward(indices)
        ctx.router_topk = router_topk
        ctx.num_local_experts = num_local_experts
        multihot_routing_map, multihot_probs = TDU.fused_topk_to_multihot(
            indices.cast(paddle.int32), probs, seqlen=indices.shape[0], topk=router_topk, num_experts=num_local_experts
        )
        return multihot_routing_map.cast(paddle.bool), multihot_probs

    @staticmethod
    def backward(ctx, grad_multihot_rouitng_map, grad_multihot_probs):
        (indices,) = ctx.saved_tensor()
        grad_probs = TDU.fused_multihot_prob_backto_topk(
            indices,
            grad_multihot_probs,
            seqlen=indices.shape[0],
            topk=ctx.router_topk,
            num_experts=ctx.num_local_experts,
        )
        return None, grad_probs


def permute(
    tokens,
    routing_map,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (paddle.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (paddle.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
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
    output_tokens.put_along_axis_(
        axis=0,
        indices=sorted_indices.unsqueeze(1).expand([-1, hidden]),
        values=permuted_tokens,
        reduce="add",
        include_self=True,
    )
    return output_tokens


class PermuteNode:
    def __init__(self, token_dispatcher, name="permute"):
        self.token_dispatcher = token_dispatcher
        self.name = name

    def forward(self, hidden_states, hidden_states_scale, indices, probs):
        self.hidden_states = hidden_states
        self.indices = indices
        self.router_topk = self.token_dispatcher._comm_manager.router_topk
        self.num_local_experts = self.token_dispatcher._comm_manager.num_local_experts
        self.token_dispatcher._comm_manager.hidden_shape_before_permute = self.hidden_states.shape

        # multi_hot_init
        multihot_routing_map, multihot_probs = TDU.fused_topk_to_multihot(
            self.indices.cast(paddle.int32),
            probs,
            seqlen=self.indices.shape[0],
            topk=self.router_topk,
            num_experts=self.num_local_experts,
        )
        self.multihot_routing_map = multihot_routing_map.cast(paddle.bool)

        # permute act
        routing_map = self.multihot_routing_map.cast(paddle.bool).T.contiguous()
        token_indices = (
            paddle.arange(self.hidden_states.shape[0]).unsqueeze(0).expand([self.multihot_routing_map.shape[1], -1])
        )
        self.sorted_indices = token_indices.masked_select(routing_map)
        output = self.hidden_states.index_select(axis=0, index=self.sorted_indices)
        # permute scale
        output_scale = hidden_states_scale.index_select(axis=0, index=self.sorted_indices)

        return (
            output,
            output_scale,
            self.sorted_indices,
            self.multihot_routing_map,
            multihot_probs,
        )

    def backward(self, output_grad, multihot_probs_grad):
        # permute_grad
        hidden_states_grad = paddle._C_ops.index_select_grad(self.hidden_states, self.sorted_indices, output_grad, 0)

        # multi_hot_grad
        probs_grad = TDU.fused_multihot_prob_backto_topk(
            self.indices.cast(paddle.int32),
            multihot_probs_grad,
            seqlen=self.indices.shape[0],
            topk=self.router_topk,
            num_experts=self.num_local_experts,
        )
        return hidden_states_grad, probs_grad
