# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) Microsoft Corporation.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
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

from typing import Any, Tuple

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed.communication import stream
from paddle.distributed.communication.group import Group

from .moe_gate_auto import PretrainedMoEGate


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
        input: Tensor,
        group: Group,
    ) -> Tensor:  # type: ignore
        """
        All-to-all communication in the group.

        Args:
            ctx (Any): Context object.
            input (Tensor): Input tensor.
            group (Group): The group object.

        Returns:
            Tensor: Output tensor.
        """

        ctx.group = group
        # return input
        if dist.get_world_size(group) <= 1:
            return input
        output = paddle.empty_like(input)
        stream.alltoall_single(output, input, None, None, group, True, True)
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
        return _AllToAll.apply(*grad_output, ctx.group)


class LocalPart(dist.LocalLayer):
    def __init__(self, out_dist_attrs, config, gate: PretrainedMoEGate):
        print("==== out_dist_attrs ====")
        print(out_dist_attrs)
        super().__init__(out_dist_attrs)
        self.config = config
        self.gate = gate

    def forward(self, hidden_state, gate_weight, used_token=None):
        # Implement Algorithm 2 from GShard paper.
        batch_size, seq_len, d_model = hidden_state.shape

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = hidden_state.reshape([-1, d_model])
        print("==== reshaped_input ===")
        print(reshaped_input)

        _, h_dim = reshaped_input.shape

        # compute gating score
        logits = F.linear(reshaped_input, gate_weight, None)

        with paddle.amp.auto_cast(False):
            scores = self.gate.gate_score_func(logits=logits)
            scores = scores.cast(paddle.get_default_dtype())

        # capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = self.topkgating(scores)
        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = self.gate.topkgating(scores)
        print("==== combine_weights ====")
        print(combine_weights)
        print("==== dispatch_mask ====")
        print(dispatch_mask)

        # self.l_aux       :
        # combine_weights  : sec
        # dispatch_mask    : sec
        # self.exp_counts  :
        dispatched_input = paddle.einsum("sec,sm->ecm", paddle.cast(dispatch_mask, hidden_state.dtype), reshaped_input)

        return dispatched_input, combine_weights, l_aux, l_zloss


class LocalCombine(dist.LocalLayer):
    def __init__(self, out_dist_attrs):
        super().__init__(out_dist_attrs)

    def forward(self, combine_weights, expert_output, dtype="float32"):
        combined_output = paddle.einsum("sec,ecm->sm", combine_weights.cast(dtype), expert_output)
        return combined_output


def get_mesh(pp_idx=0):
    """
    获得pp_idx的mesh
    """
    mesh = dist.fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp", pp_idx)
    return mesh


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

        print(f"moe_num_experts:{moe_num_experts}")
        self.moe_num_experts = moe_num_experts
        self.capacity = capacity
        self.expert_parallel_degree = 1

        self.all_to_all_dropout = all_to_all_dropout
        self.enable_recompute = False

        self.experts = nn.LayerList([])
        for i in range(self.moe_num_experts):
            self.experts.append(expert_class(**expert_kwargs))

        self.moe_num_experts_per_device = self._parse_moe_expert_parallel(
            self.moe_num_experts, self.expert_parallel_degree
        )
        self.moe_group = None
        self.gate = gate
        self.gate.group = self.moe_group
        self.is_dummy_moe = True
        self._post_init()

        mesh = get_mesh()
        local_out_dist_attrs = [
            (mesh, [dist.Shard(1)]), # dispatched_input [e,c,h]
            (mesh, [dist.Shard(0)]), # combine_weights [s,e,c]
            (mesh, [dist.Partial()]), # l_aux, scalar
            (mesh, [dist.Partial()]), # l_zloss, scalar
        ]
        self.local_computes = LocalPart(local_out_dist_attrs, config, gate)

        local_combine_dist_attrs = [
            (mesh, [dist.Shard(0)])
        ]
        self.local_combine = LocalCombine(local_combine_dist_attrs)

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

    def expert_forward(self, dispatched_input):
        expert_outputs = []
        chunks = dispatched_input.unbind(1)
        for chunk, expert in zip(chunks, self.experts):
            chunk = chunk.contiguous()
            expert_outputs += [expert(chunk)]
        expert_output = paddle.stack(expert_outputs, axis=1)  # [ecm]
        return expert_output

    def forward(
        self,
        hidden_state: paddle.Tensor,
        used_token: paddle.Tensor = None,
    ):
        """_summary_

        Args:
            input (_type_): _description_
            used_token

        Returns:
            _type_: _description_
        """
        # Implement Algorithm 2 from GShard paper.
        batch_size, seq_len, d_model = hidden_state.shape

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        # reshaped_input = hidden_state.reshape([-1, d_model])
        # reshaped_input = dist.reshard(reshaped_input, reshaped_input.process_mesh, [dist.Replicate(), dist.Replicate()])
        # print("==== reshaped_input ====")
        # print(reshaped_input)

        # capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = self.gate(reshaped_input)
        # print("==== combine_weights ====")
        # print(combine_weights)

        # # self.l_aux       :
        # # combine_weights  : sec
        # # dispatch_mask    : sec
        # # self.exp_counts  :
        # dispatched_input = paddle.einsum("sec,sm->ecm", paddle.cast(dispatch_mask, hidden_state.dtype), reshaped_input)
        # print("==== dispatched_input ====")
        # print(dispatched_input)

        print("==== hidden_state ====")
        print(hidden_state)
        dispatched_input, combine_weights, l_aux, l_zloss = self.local_computes(hidden_state, self.gate.weight, used_token=used_token)

        # dispatched_input = dist.reshard(dispatched_input, get_mesh(), [dist.Shard(0)])
        # if self.expert_parallel_degree > 1:
        #     dispatched_input = _AllToAll.apply(dispatched_input, self.moe_group)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(
            [self.expert_parallel_degree, self.moe_num_experts_per_device, -1, d_model]
        )
        expert_output = self.expert_forward(dispatched_input)
        # Re-shape before drop_tokens: gecm -> ecm
        expert_output = expert_output.reshape(
            [self.expert_parallel_degree * self.moe_num_experts_per_device, -1, d_model]
        )
        print("==== expert_output ===")
        print(expert_output)

        expert_output = dist.reshard(expert_output, get_mesh(), [dist.Shard(1)])
        print("==== expert_output after reshard ====")
        print(expert_output)
        # if self.expert_parallel_degree > 1:
        #     expert_output = _AllToAll.apply(expert_output, self.moe_group)

        print("==== combine_weights ====")
        print(combine_weights)
        # combine withe expert weights
        # Einsum infermeta has not supported auto parallel dist tensor,
        # so use local layer here.
        # combined_output = paddle.einsum("sec,ecm->sm", combine_weights.cast(hidden_state[0].dtype), expert_output)
        combined_output = self.local_combine(combine_weights, expert_output, dtype=hidden_state[0].dtype)

        a = combined_output.reshape(hidden_state.shape)
        print("==== a ====")
        print(a)

        return a, l_aux, l_zloss
