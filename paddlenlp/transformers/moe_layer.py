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

import paddle
import paddle.distributed as dist
from paddle import nn

from .moe_gate import PretrainedMoEGate


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

        if is_fleet_init and dist.get_world_size() > 1 and moe_group == "data":
            self.moe_group = dist.fleet.get_hybrid_communicate_group().get_data_parallel_group()
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
        self._post_init()

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

    def expert_forward(self, dispatched_input, exp_token_idx):
        true_experts = self.experts[
            self.moe_rank * self.moe_num_experts_per_device : (self.moe_rank + 1) * self.moe_num_experts_per_device
        ]
        expert_outputs = []

        for idx in range(len(dispatched_input)):
            # print(dispatched_input[idx])
            # print(exp_token_idx[idx])
            # (LiuTing) can use paddle.stack here.
            expert_outputs.append(
                true_experts[idx % self.moe_num_experts_per_device](dispatched_input[idx])
                if exp_token_idx[idx] is not None
                else dispatched_input[idx]
            )

        return expert_outputs

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
        reshaped_input = hidden_state.reshape([-1, d_model])

        # self.l_aux       :
        # gates_masked  : se
        # token_priority    : se
        # self.exp_counts  :
        capacity, gates_masked, token_priority, exp_counts, l_aux, l_zloss = self.gate(hidden_state)

        dispatched_input = []
        # print("token p: ", token_priority)
        # print("token pick exp num: ", (token_priority>=0).astype('bfloat16').sum(axis=1))
        # print("less pick exp: ", (token_priority>=0).astype('bfloat16').sum(axis=0).min())
        # print("gates masked: ", gates_masked)
        exp_token_idx = []
        for e_i in range(self.moe_num_experts):
            expert_tokens_idx = (token_priority[:, e_i] >= 0).nonzero().squeeze(-1)
            # (LiuTing) this expert not deal with any token.
            if expert_tokens_idx.shape[0] == 0:
                exp_token_idx.append(None)
                dispatched_input.append(None)
            else:
                exp_token_idx.append(expert_tokens_idx)
                dispatched_input.append(paddle.gather(reshaped_input, exp_token_idx[e_i], axis=0))

        if self.expert_parallel_degree > 1:
            output = []
            dist.alltoall(output, dispatched_input, self.moe_group)
            dispatched_input = output

        expert_output = self.expert_forward(dispatched_input, exp_token_idx)
        if self.expert_parallel_degree > 1:
            output = []
            dist.alltoall(output, expert_output, self.moe_group)
            expert_output = output

        # reformat output
        a = paddle.zeros_like(reshaped_input)

        for e_i in range(self.moe_num_experts):
            if exp_token_idx[e_i] is not None:
                updated = expert_output[e_i] * gates_masked[exp_token_idx[e_i].unsqueeze(-1), e_i]
                # print("e_i: ", e_i)
                # print("act: ", token_priority[:, e_i])
                # print("expert_output: ", expert_output[e_i])
                # print("exp token idxx: ", exp_token_idx[e_i])
                # print("gate weight: ", gates_masked[exp_token_idx[e_i].unsqueeze(-1), e_i])
                # print("gates value: ", updated)
                a[exp_token_idx[e_i]] += updated.astype(a.dtype)

        # print("moe output: ", a, a.mean(), a.max(), a._md5sum())
        return a, l_aux, l_zloss
