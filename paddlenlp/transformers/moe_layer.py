# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) Microsoft Corporation.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
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
from __future__ import annotations

import os
from typing import Any, List, Tuple

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.distributed.communication.group import Group

from ..utils.log import logger
from .fp8_utils import ExpertsGroupGemmNode, ExpertsNode, kitchen_quant
from .fused_a2a import CombineNode, DispatchNode
from .moe_gate import PretrainedMoEGate
from .moe_utils import PermuteNode, UnPermuteNode, UnZipNode, ZipNode
from .token_dispatcher import MoEFlexTokenDispatcher, PreDispatchNode

try:
    import kitchen
except:
    pass

DSV3_USE_FP8_GEMM = os.getenv("DSV3_USE_FP8_GEMM", "False").lower() == "true"

DSV3_USE_FP8_GROUP_GEMM = os.getenv("DSV3_USE_FP8_GROUP_GEMM", "False").lower() == "true"


def dispatching(x, dispatch_mask, scatter_index, num_experts, capacity):
    """
    Rearranges the input tensor `x` based on gate results, truncates it according to the specified capacity, and performs padding.

    Args:
        x (Tensor)[Seq, Dim]: The input tensor.
        dispatch_mask (List[Tensor[Seq, 1], Tensor[Seq, 1]]): A list of dispatch masks.
        scatter_index (Union[List[Tensor[Seq,], Tensor[Seq]], Tensor[Seq, 2]]): A list or tensor representing scatter indices.
        num_experts (int): The number of experts.
        capacity (int): The capacity size.

    Returns:
        Tensor [Expert*Capacity, Dim]: The output tensor after dispatching.
    """
    output = None
    orig_dtype = x.dtype
    if isinstance(scatter_index, paddle.Tensor):
        scatter_index = scatter_index.unbind(1)
    for i_scatter_index, i_dispatch_mask in zip(scatter_index, dispatch_mask):
        init_output = paddle.zeros([num_experts * capacity, x.shape[-1]], dtype="float32")
        updates = x * i_dispatch_mask.cast(x.dtype)
        if output is None:
            output = paddle.scatter(
                init_output,
                i_scatter_index,
                updates,
                overwrite=False,
            )
        else:
            output = output + paddle.scatter(
                init_output,
                i_scatter_index,
                updates,
                overwrite=False,
            )
        if output.dtype != orig_dtype:
            output = output.cast(orig_dtype)
    return output


def combining(x, combine_weights, scatter_index):
    """
    Performs combination and aggregation operations on the input matrix.

    Args:
        x: Tensor[num_experts * capacity, dim] - The input matrix to be processed, where the last dimension represents the number of features.
        combine_weights: Union[List[Tensor[seq, 1], Tensor[seq, 1]], Tensor[seq, 2, 1]] - A list or tensor containing combination weights for each feature.
        scatter_index: Union[List[Tensor[seq], Tensor[seq]], Tensor[seq, 2]] - A tuple of indices indicating which elements are to be aggregated, where the first element is the row index and the second element is the column index.

    Returns:
        Tensor: The output matrix after combination and aggregation, with a shape of [n, dim * num_features], where n is the number of samples in the input matrix.
    """

    dim = x.shape[-1]
    if isinstance(scatter_index, (list, tuple)):
        scatter_index = paddle.concat([i.unsqueeze([-1]) for i in scatter_index], -1)
    scatter_index = scatter_index.reshape([-1])
    num_k = len(combine_weights) if isinstance(combine_weights, (list, tuple)) else combine_weights.shape[-1]
    x = paddle.gather(x, scatter_index).reshape([-1, num_k, dim])  # [seq,2,dim]
    if isinstance(combine_weights, (list, tuple)):
        combine_weights = paddle.concat(combine_weights, -1).unsqueeze([1])
    return paddle.matmul(combine_weights, x).squeeze(1)  # [seq,1,2] @ [seq,2,dim] -> [seq,1,dim]


class _AllToAll(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx: Any,
        output_shape: List,
        input: Tensor,
        out_split_sizes: List = None,
        in_split_sizes: List = None,
        group: Group = None,
    ) -> Tensor:  # type: ignore
        """
        All-to-all communication in the group.
        Args:
            ctx (Any): Context object.
            output_shape (List): Output shape.
            input (Tensor): Input tensor.
            out_split_sizes (List): Output split sizes.
            in_split_sizes (List): Input split sizes.
            group (Group): The group object.
        Returns:
            Tensor: Output tensor.
        """

        ctx.group = group
        ctx.input_shape = input.shape
        ctx.out_split_sizes = out_split_sizes
        ctx.in_split_sizes = in_split_sizes

        # return input
        if dist.get_world_size(group) <= 1:
            return input

        output = paddle.empty(output_shape, dtype=input.dtype)
        task = dist.alltoall_single(
            output,
            input,
            out_split_sizes=out_split_sizes,
            in_split_sizes=in_split_sizes,
            sync_op=False,
            group=group,
        )
        task.wait()

        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[Tensor]:
        """
        Aggregates gradient information from all input tensors into a single tensor.
        Args:
            ctx (Any): The context object used to store information that needs to be passed.
            *grad_output (Tensor): A list of input tensors whose gradients are to be aggregated.
        Returns:
            Tuple[Tensor]: A tuple containing a tensor that holds the gradients of all input tensors.
        """
        # return grad_output
        return _AllToAll.apply(ctx.input_shape, *grad_output, ctx.in_split_sizes, ctx.out_split_sizes, ctx.group)


class MoELayer(nn.Layer):
    def __init__(
        self,
        config,
        moe_num_experts: int,
        expert_class: nn.Layer,
        expert_kwargs: dict,
        gate: PretrainedMoEGate,
        capacity: int = 1.0,
        moe_group: str = "data",
        all_to_all_dropout=0.0,
    ):
        super().__init__()

        self.config = config

        self.moe_num_experts = moe_num_experts
        self.capacity = capacity

        try:
            dist.fleet.get_hybrid_communicate_group()
            is_fleet_init = True
        except AttributeError:
            is_fleet_init = False

        if is_fleet_init and dist.get_world_size() > 1:
            if moe_group == "data":
                self.moe_group = dist.fleet.get_hybrid_communicate_group().get_data_parallel_group()
            elif moe_group == "expert":
                self.moe_group = dist.fleet.get_hybrid_communicate_group().expert_parallel_group
            self.moe_rank = dist.get_rank(self.moe_group)
            self.moe_rank = 0 if self.moe_rank < 0 else self.moe_rank
            self.expert_parallel_degree = dist.get_world_size(self.moe_group)
            self.expert_parallel_degree = 1 if self.expert_parallel_degree < 0 else self.expert_parallel_degree
            self.moe_num_experts_per_device = self._parse_moe_expert_parallel(
                self.moe_num_experts, self.expert_parallel_degree
            )
            self.is_dummy_moe = False if self.expert_parallel_degree > 1 else True
        else:
            # when moe_group is dummy, we don't need to use all_to_all
            self.moe_group = None
            self.moe_rank = 0
            self.expert_parallel_degree = 1
            self.moe_num_experts_per_device = self.moe_num_experts
            self.is_dummy_moe = True

        self.all_to_all_dropout = all_to_all_dropout
        self.enable_recompute = False

        self.experts = nn.LayerList([])
        for i in range(self.moe_num_experts):
            if i // self.moe_num_experts_per_device == self.moe_rank:
                self.experts.append(expert_class(**expert_kwargs))
            else:
                self.experts.append(None)

        self.gate = gate
        self.gate.group = self.moe_group
        # for flex token moe layer
        self.router = gate
        self.ep_size = dist.get_world_size(self.moe_group)
        self.moe_router_topk = gate.top_k
        self.num_local_experts = moe_num_experts // self.ep_size
        self.token_dispatcher = MoEFlexTokenDispatcher(
            self.num_local_experts, self.moe_router_topk, self.moe_num_experts, self.moe_group
        )
        self.token_drop_steps = config.token_drop_steps
        self.using_flex_token = False
        self._post_init()

    def update_flex_token(self):
        from paddlenlp.transformers.deepseek_v2 import get_global_step

        if (not self.config.using_flex_token) or (get_global_step() < self.token_drop_steps):
            self.using_flex_token = False
            self.router.using_flex_token = False
        else:
            if not self.using_flex_token:
                logger.info("Changing to flex token moe mode")
            self.using_flex_token = True
            self.router.using_flex_token = True

    def _parse_moe_expert_parallel(self, moe_num_experts, expert_parallel_degree):
        assert (
            moe_num_experts >= expert_parallel_degree
        ), f"expert moe_num_experts={moe_num_experts} >= moe_world_size={expert_parallel_degree}"
        assert (
            moe_num_experts % expert_parallel_degree == 0
        ), f"expert moe_num_experts={moe_num_experts} % moe_world_size={expert_parallel_degree} == 0"
        moe_num_experts_per_device = moe_num_experts // expert_parallel_degree
        return moe_num_experts_per_device

    def _post_init(self):
        for p in self.gate.parameters():
            p.is_gate = True

        for k in self.experts:
            if k is not None:
                for p in k.parameters():
                    p.expert = not self.is_dummy_moe
                    p.no_sync = not self.is_dummy_moe
                    # logger.info(f"expert param={p.name}, no-sync={p.no_sync}")

    def forward(self, hidden_states: paddle.Tensor):
        self.update_flex_token()

        if self.using_flex_token:
            return self.forward_flex_token(hidden_states)
        else:
            return self.forward_drop_token(hidden_states)

    def forward_drop_token(
        self,
        hidden_state: paddle.Tensor,
    ):
        """MoE Layer forward function
            1. Gate Forward.
            2. Dispatch export.
            3. Experts Forward.

        Args:
            hidden_state: MoE Layer input

        Returns:
            final_out: MoE Layer main output.
            l_aux: MoE auxiliary loss.  l_zloss: MoE z loss."""
        batch_size, seq_len, d_model = hidden_state.shape

        reshaped_input = hidden_state.reshape([-1, d_model])

        # self.l_aux       :
        # topk_weight  : se
        # topk_ids    : sk
        # token_priority    : se
        # self.exp_counts  :
        capacity, topk_weight, topk_ids, token_priority, l_aux, l_zloss = self.gate(hidden_state)

        """MoE expert dispatch from: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py"""
        cnts = paddle.zeros([topk_ids.shape[0], len(self.experts)], dtype=topk_ids.dtype)
        cnts = cnts.put_along_axis(topk_ids, 1, axis=1)

        tokens_per_expert = cnts.sum(axis=0)
        idxs = topk_ids.reshape([topk_ids.shape[0] * topk_ids.shape[1]]).argsort()
        sorted_tokens = reshaped_input[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.detach()
        sorted_tokens_shape = sorted_tokens.shape

        if self.expert_parallel_degree > 1:
            tokens_per_ep_rank = tokens_per_expert.reshape([self.expert_parallel_degree, -1]).sum(axis=1)
            tokens_per_expert_group = _AllToAll.apply(
                [tokens_per_expert.shape[0]], tokens_per_expert, group=self.moe_group
            )
            output_splits = (
                tokens_per_expert_group.reshape([self.expert_parallel_degree, -1]).sum(axis=1).cpu().tolist()
            )
            input_split_sizes = tokens_per_ep_rank.cpu().tolist()
            gathered_tokens = _AllToAll.apply(
                [tokens_per_expert_group.sum(axis=0).cpu().item(), sorted_tokens.shape[1]],
                sorted_tokens,
                out_split_sizes=output_splits,
                in_split_sizes=input_split_sizes,
                group=self.moe_group,
            )

            tokens_per_expert_post_gather = tokens_per_expert_group.reshape(
                [self.expert_parallel_degree, self.moe_num_experts_per_device]
            ).sum(axis=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.moe_num_experts_per_device
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.moe_rank * self.moe_num_experts_per_device]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = paddle.concat(outputs, axis=0) if len(outputs) > 0 else paddle.to_tensor(0, dtype=sorted_tokens.dtype)
        if self.expert_parallel_degree > 1:
            new_x = paddle.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = _AllToAll.apply(
                sorted_tokens_shape,
                new_x,
                out_split_sizes=input_split_sizes,
                in_split_sizes=output_splits,
                group=self.moe_group,
            )
            outs = gathered_tokens

        new_x = paddle.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.reshape(topk_ids.shape + [-1])
            .astype(topk_weight.dtype)
            .multiply_(topk_weight.unsqueeze(-1))
            .multiply_(token_priority.unsqueeze(-1))
            .sum(axis=1)
            .astype(new_x.dtype)
            .reshape([batch_size, seq_len, -1])
        )

        return final_out, l_aux, l_zloss

    def forward_flex_token(self, hidden_states: paddle.Tensor):
        _, _, d_model = hidden_states.shape
        # reshaped_input = hidden_states.reshape([-1, d_model])
        probs, routing_map, l_aux, l_zloss = self.router(hidden_states)

        if DSV3_USE_FP8_GEMM:
            output = FusionMoe.apply(hidden_states, probs, routing_map, self)
        else:
            (
                dispatched_input,
                token_permuted_indices,
                prob_permuted_indices,
                dispatched_probs,
            ) = self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
            expert_output = self.expert_forward(dispatched_input)
            output, _ = self.token_dispatcher.token_unpermutation(
                expert_output, token_permuted_indices, prob_permuted_indices, dispatched_probs, None
            )

        return output, l_aux, l_zloss

    def get_tokens_per_expert(self):
        return self.token_dispatcher._comm_manager.tokens_per_expert_list

    def set_tokens_per_expert(self, tokens_per_expert_list):
        self.token_dispatcher._comm_manager.tokens_per_expert_list = tokens_per_expert_list

    def expert_forward(self, dispatched_input):
        outputs = []
        chunks = paddle.split(dispatched_input, num_or_sections=self.get_tokens_per_expert(), axis=0)
        for i, chunk in enumerate(chunks):
            chunk = chunk.contiguous()
            # assert chunk.shape[0] != 0, "Cannot dispatch empty input"
            expert = self.experts[i + self.moe_rank * self.moe_num_experts_per_device]
            outputs += [expert(chunk)]

        return paddle.concat(outputs, axis=0)

    def pre_dispatch_compute(self, hidden_states):
        _, _, d_model = hidden_states.shape
        probs, routing_map, l_aux, l_zloss = self.router(hidden_states)
        hidden_states, token_indices, token_probs = self.token_dispatcher.pre_dispatch(
            hidden_states, probs, routing_map
        )
        return l_aux, l_zloss, hidden_states, token_indices, token_probs

    def post_dispatch_compute(self, hidden_states, dispatched_indices, dispatched_probs):
        (global_input_tokens, token_permuted_indices, prob_permuted_indices) = self.token_dispatcher.post_dispatch(
            hidden_states, dispatched_indices
        )
        return (global_input_tokens, token_permuted_indices, prob_permuted_indices)

    def pre_combine_compute(self, hidden_states, token_permuted_indices, prob_permuted_indices, dispatched_probs):
        hidden_states = self.token_dispatcher.pre_combine(
            hidden_states, token_permuted_indices, prob_permuted_indices, dispatched_probs
        )
        return hidden_states

    def post_combine_compute(self, hidden_states):
        hidden_states = self.token_dispatcher.post_combine(hidden_states)
        return hidden_states


class Fp8DispatchQuantNode:
    def __init__(self, token_dispatcher, name="fp8_dispatch_quant_node"):
        self.token_dispatcher = token_dispatcher
        self.pre_dispatch_node = PreDispatchNode(token_dispatcher)
        self.name = name

    @paddle.no_grad()
    def forward(self, hidden_states, probs, routing_map):
        # reshape
        self.token_dispatcher.hidden_shape = hidden_states.shape
        hs_2d = hidden_states.view([-1, self.token_dispatcher.hidden_shape[-1]])

        # quant
        hs_fp8, hs_scale = kitchen_quant(
            hs_2d, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )

        # pre_dispatch
        token_indices, token_probs = self.pre_dispatch_node.forward(routing_map, probs)

        self.hidden_states_shape = hidden_states.shape
        hs_fp8.stop_gradient = False
        token_probs.stop_gradient = False
        return hs_fp8, hs_scale, token_indices, token_probs

    @paddle.no_grad()
    def backward(self, hs_fp8_grad, token_probs_grad):
        # predispatch grad
        probs_grad = self.pre_dispatch_node.backward(token_probs_grad)
        token_probs_grad._record_stream()

        # reshape_grad
        hs_grad = paddle.reshape(hs_fp8_grad, self.hidden_states_shape)
        hs_fp8_grad._record_stream()

        return hs_grad, probs_grad, None


class Fp8DispatchNode:
    def __init__(self, token_dispatcher, name="fp8_dispatch_node"):
        self.token_dispatcher = token_dispatcher
        self.dispatch_act_node = DispatchNode(token_dispatcher)
        self.name = name

    @paddle.no_grad()
    def forward(
        self,
        hs_fp8,
        hs_scale,
        token_indices,
        token_probs,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        # dispatch
        (hs_fp8_dispatched, hs_scale_dispatched), dispatched_probs, states = self.dispatch_act_node.forward(
            (hs_fp8, hs_scale),
            token_indices,
            token_probs,
            self.token_dispatcher._comm_manager.num_experts,
            self.token_dispatcher._comm_manager.group,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.token_dispatcher._comm_manager.handle = states["handle"]
        self.token_dispatcher._comm_manager.tokens_per_expert = states["tokens_per_expert"]
        dispatched_indices = states["dispatched_indices"]

        hs_fp8_dispatched.stop_gradient = False
        dispatched_probs.stop_gradient = False
        return hs_fp8_dispatched, hs_scale_dispatched, dispatched_indices, dispatched_probs

    @paddle.no_grad()
    def backward(self, hs_fp8_dispatched_grad, dispatched_probs_grad, previous_event=None, async_finish=False):
        # dispatch grad
        hs_fp8_grad, _, token_probs_grad = self.dispatch_act_node.backward(
            hs_fp8_dispatched_grad,
            dispatched_probs_grad,
            previous_event=previous_event,
            async_finish=async_finish,
        )
        return hs_fp8_grad, token_probs_grad


class Fp8CombineNode:
    def __init__(self, token_dispatcher, name="fp8_combine_node"):
        self.token_dispatcher = token_dispatcher
        self.combine_node = CombineNode(token_dispatcher)
        self.name = name

    @paddle.no_grad()
    def forward(self, hidden_states_out, previous_event=None, async_finish=False):
        # combine
        output_combie = self.combine_node.forward(
            hidden_states_out,
            self.token_dispatcher._comm_manager.group,
            self.token_dispatcher._comm_manager.handle,
            previous_event=previous_event,
            async_finish=async_finish,
        )
        output_combie.stop_gradient = False
        return output_combie

    @paddle.no_grad()
    def backward(self, output_combie_grad_fp8, output_combie_grad_scale, previous_event=None, async_finish=False):
        # combine grad -> fp8
        (hidden_states_out_grad, hidden_states_out_grad_scale) = self.combine_node.backward(
            (output_combie_grad_fp8, output_combie_grad_scale),
            previous_event=previous_event,
            async_finish=async_finish,
        )
        return hidden_states_out_grad, hidden_states_out_grad_scale


class Fp8CombineQuantNode:
    def __init__(self, token_dispatcher, name="fp8_combine_quant_node"):
        self.token_dispatcher = token_dispatcher
        self.name = name

    @paddle.no_grad()
    def forward(self, output_combie):
        # post combine
        output = output_combie.reshape(self.token_dispatcher.hidden_shape)
        output_combie._record_stream()
        self.output_combie_shape = output_combie.shape
        output.stop_gradient = False
        return output

    @paddle.no_grad()
    def backward(self, output_grad):
        # post combine grad
        output_combie_grad = paddle.reshape(output_grad, self.output_combie_shape)

        # output_combie_grad quant to fp8
        output_combie_grad_fp8, output_combie_grad_scale = kitchen_quant(
            output_combie_grad, backend=kitchen.ops.Backend.CUTLASS, is_1d_scaled=True, return_transpose=False
        )
        return output_combie_grad_fp8, output_combie_grad_scale


class MlpNode:
    def __init__(self, custom_map, name="mlp_node"):
        self.token_dispatcher = custom_map.token_dispatcher
        self.experts = custom_map.experts
        self.permute_node = PermuteNode(self.token_dispatcher)
        self.experts_node = ExpertsNode(self.experts, custom_map)
        self.experts_group_gemm_node = ExpertsGroupGemmNode(self.experts, custom_map)
        self.unpermute_node = UnPermuteNode(self.token_dispatcher)
        self.name = name
        self.unzip_node = UnZipNode(self.token_dispatcher)
        self.zip_node = ZipNode(self.token_dispatcher)
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.total_unzipped_tokens_num = None
        self.unzipped_expert_idx = None
        self.unzipped_scale = None
        self.unzipped_tokens = None
        self.tokens_per_expert = None
        self.router_topk = None

    def reset_statue(self):
        self.token_permuted_indices = None
        self.dispatched_probs = None
        self.total_unzipped_tokens_num = None
        self.unzipped_expert_idx = None
        self.dispatched_indices = None
        self.unzipped_scale = None
        self.unzipped_tokens = None
        self.tokens_per_expert = None
        self.router_topk = None

    @paddle.no_grad()
    def forward(self, hs_fp8_dispatched, hs_scale_dispatched, dispatched_indices, dispatched_probs):
        self.tokens_per_expert = self.token_dispatcher._comm_manager.tokens_per_expert
        self.router_topk = self.token_dispatcher._comm_manager.router_topk
        if DSV3_USE_FP8_GROUP_GEMM:
            # 1 unzip
            dispatched_indices = dispatched_indices.to(paddle.int32)

            self.dispatched_indices = dispatched_indices
            (unzipped_tokens, unzipped_scale, zipped_expertwise_rowmap, unzipped_probs,) = self.unzip_node.forward(
                hs_fp8_dispatched,
                hs_scale_dispatched,
                dispatched_indices,
                dispatched_probs,
                topk=self.router_topk,
                num_experts=4,
                max_tokens=max(self.tokens_per_expert),
            )
            hs_fp8_dispatched._record_stream()
            hs_scale_dispatched._record_stream()
            dispatched_indices._record_stream()
            dispatched_probs._record_stream()

            # 2 experts
            tokens_per_expert_list = []
            for idx in range(len(self.tokens_per_expert)):
                tokens_per_expert_list.append(
                    paddle.full(shape=[1], fill_value=self.tokens_per_expert[idx], dtype="int32")
                )
            tokens_per_expert = paddle.concat(tokens_per_expert_list)
            # expected_m = int(max(tokens_per_expert))
            expert_out = self.experts_group_gemm_node.forward(
                unzipped_tokens, unzipped_scale, unzipped_probs, tokens_per_expert
            )
            self.unzipped_tokens = unzipped_tokens
            self.unzipped_scale = unzipped_scale

            # 3 zip
            # expert是3维，reshape成2维
            expert_out = expert_out.reshape([-1, expert_out.shape[-1]])
            expert_out_zipped = self.zip_node.forward(
                expert_out,
                zipped_expertwise_rowmap,
                dispatched_indices,
                unzipped_probs,
                total_zipped_tokens=hs_fp8_dispatched.shape[0],
                num_experts=4,
            )
            self.dispatched_probs = dispatched_probs
            expert_out_zipped.stop_gradient = False
            return expert_out_zipped
        else:
            # permute
            (
                hs_out,
                hs_scale_out,
                token_permuted_indices,
                prob_permuted_indices,
            ) = self.permute_node.forward(hs_fp8_dispatched, hs_scale_dispatched, dispatched_indices)
            hs_fp8_dispatched._record_stream()
            hs_scale_dispatched._record_stream()
            dispatched_indices._record_stream()

            # experts
            expert_out = self.experts_node.forward(
                hs_out, hs_scale_out, self.token_dispatcher._comm_manager.tokens_per_expert
            )

            # unpermute
            hidden_states_out = self.unpermute_node.forward(
                expert_out, token_permuted_indices, prob_permuted_indices, dispatched_probs
            )
            dispatched_probs._record_stream()
            self.dispatched_probs = dispatched_probs
            self.token_permuted_indices = token_permuted_indices
            hidden_states_out.stop_gradient = False
            return hidden_states_out

    @paddle.no_grad()
    def backward(self, hidden_states_out_grad, hidden_states_out_grad_scale):
        if DSV3_USE_FP8_GROUP_GEMM:
            # zip_grad
            unzipped_grad, unzipped_scale_grad = self.zip_node.backward(
                hidden_states_out_grad,
                hidden_states_out_grad_scale,
                self.dispatched_indices,
                self.dispatched_probs,
                top_k=self.router_topk,
                num_experts=4,
                max_tokens=max(self.tokens_per_expert),
            )
            hidden_states_out_grad._record_stream()
            hidden_states_out_grad_scale._record_stream()

            # expert_grad
            tokens_per_expert_list = []
            for idx in range(len(self.tokens_per_expert)):
                tokens_per_expert_list.append(
                    paddle.full(shape=[1], fill_value=self.tokens_per_expert[idx], dtype="int32")
                )
            tokens_per_expert = paddle.concat(tokens_per_expert_list)
            expected_m = int(max(self.tokens_per_expert))
            expert_out, probs_grad = self.experts_group_gemm_node.backward(
                unzipped_grad, unzipped_scale_grad, tokens_per_expert, self.dispatched_indices, expected_m
            )
            # unzip_grad
            hs_fp8_dispatched_grad, dispatched_probs_grad = self.unzip_node.backward(
                expert_out, hidden_states_out_grad, probs_grad, self.dispatched_indices
            )
            self.reset_statue()
            return hs_fp8_dispatched_grad, dispatched_probs_grad

        else:
            # unpermute grad -> fp8
            expert_out_grad, dispatched_probs_grad = self.unpermute_node.backward(
                hidden_states_out_grad, hidden_states_out_grad_scale
            )
            hidden_states_out_grad_scale_grad = paddle.gather(
                hidden_states_out_grad_scale, self.token_permuted_indices
            )

            hidden_states_out_grad._record_stream()
            hidden_states_out_grad_scale._record_stream()

            # expert_grad
            hs_out_grad = self.experts_node.backward(expert_out_grad, hidden_states_out_grad_scale_grad)

            # permute_grad
            hs_fp8_dispatched_grad = self.permute_node.backward(hs_out_grad, self.dispatched_probs)
            self.reset_statue()
            return hs_fp8_dispatched_grad, dispatched_probs_grad


class FusionMoeNode:
    def __init__(self, custom_map, name="fusion_moe_node"):
        self.token_dispatcher = custom_map.token_dispatcher

        self.dispatch_quant_node = Fp8DispatchQuantNode(self.token_dispatcher)
        self.dispatch_node = Fp8DispatchNode(self.token_dispatcher)
        self.mlp_node = MlpNode(custom_map)
        self.combine_node = Fp8CombineNode(self.token_dispatcher)
        self.combine_quant_node = Fp8CombineQuantNode(self.token_dispatcher)
        self.name = name

    @paddle.no_grad()
    def forward(self, hidden_states, probs, routing_map):
        hs_fp8, hs_scale, token_indices, token_probs = self.dispatch_quant_node.forward(
            hidden_states, probs, routing_map
        )
        hs_fp8_dispatched, hs_scale_dispatched, dispatched_indices, dispatched_probs = self.dispatch_node.forward(
            hs_fp8, hs_scale, token_indices, token_probs
        )
        hidden_states_out = self.mlp_node.forward(
            hs_fp8_dispatched, hs_scale_dispatched, dispatched_indices, dispatched_probs
        )
        output_combie = self.combine_node.forward(hidden_states_out)
        output = self.combine_quant_node.forward(output_combie)
        output.stop_gradient = False
        return output

    @paddle.no_grad()
    def backward(self, output_grad):
        output_combie_grad_fp8, output_combie_grad_scale = self.combine_quant_node.backward(output_grad)
        hidden_states_out_grad, hidden_states_out_grad_scale = self.combine_node.backward(
            output_combie_grad_fp8, output_combie_grad_scale
        )
        hs_fp8_dispatched_grad, dispatched_probs_grad = self.mlp_node.backward(
            hidden_states_out_grad, hidden_states_out_grad_scale
        )
        hs_fp8_grad, token_probs_grad = self.dispatch_node.backward(hs_fp8_dispatched_grad, dispatched_probs_grad)
        hs_grad, probs_grad, routing_map_grad = self.dispatch_quant_node.backward(hs_fp8_grad, token_probs_grad)
        return hs_grad, probs_grad, routing_map_grad


class FusionMoe(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, hidden_states, probs, routing_map, custom_map):
        ctx.node = FusionMoeNode(custom_map)
        return ctx.node.forward(hidden_states, probs, routing_map)

    @staticmethod
    def backward(ctx, output_grad):
        return ctx.node.backward(output_grad)
