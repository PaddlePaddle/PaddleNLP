# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from collections import namedtuple
from contextlib import contextmanager
from typing import Any, List, Tuple

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.communication import stream
from paddle.distributed.communication.group import Group
from paddle.distributed.fleet.utils import recompute

from ..utils.log import logger

GateOutput = namedtuple(
    "GateOutput",
    [
        "aux",
        "z",
        "logits",
    ],
)


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


class MoELayer(nn.Layer):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)

        moe = MoELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (paddle.nn.Layer):
            gate network
        expert (paddle.nn.LayerList):
            expert network, LayerList 长度是 per_device 上的 expert 数。
        group (paddle.ProgressGroup)
        recompute: 启用MOE内recomupte
    Returns:
        output
        combine_weight
        router-loss
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        group: Group = None,
        recompute=False,
        all_to_all_dropout=0.0,
        moe_num_experts=2,
    ):
        super().__init__()
        self.gate = gate
        self.layer_idx = layer_idx
        self.recompute = recompute
        logger.info(f"using moe recompute={recompute}")
        for p in self.gate.parameters():
            p.is_gate = True
        if type(experts) == nn.LayerList:
            self.experts = experts
        else:
            logger.info(f"using fused experts, type={type(experts)}")
            self.experts = nn.LayerList([experts])
        self.group = group
        self.all_to_all_dropout = all_to_all_dropout
        is_dummy_moe = dist.get_world_size(group) == 1 or dist.get_world_size(group) == -1

        for k in experts:
            if k is not None:
                for p in k.parameters():
                    p.expert = not is_dummy_moe
                    p.no_sync = not is_dummy_moe
                    # logger.info(f"expert param={p.name}, no-sync={p.no_sync}")

        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(self.group)
        if self.world_size < 1:
            self.world_size = 1
        if self.rank < 0:
            self.rank = 0

        self.num_local_experts = moe_num_experts // self.world_size

    def forward(self, input):
        true_experts = self.experts[self.rank * self.num_local_experts : (self.rank + 1) * self.num_local_experts]
        if input.ndim == 3:
            orig_shape = input.shape
            reshaped_input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert len(input.shape) == 2, f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"

        # Implement Algorithm 2 from GShard paper.
        seqlen, d_model = input.shape

        # Reshape into S tokens by dropping sequence dimension.
        # reshaped_input = input.reshape(-1, d_model)
        # assert reshaped_input.shape[0] % len(self.experts) == 0, \
        # f'num tokens must be order of number of local experts, {input[0].shape[0]} vs {len(self.experts)}'
        def fwdfn(dispatched_input):
            expert_outputs = []
            chunks = dispatched_input.unbind(1)
            assert len(chunks) == len(true_experts), (len(chunks), len(true_experts))
            for chunk, expert in zip(chunks, true_experts):
                chunk = chunk.contiguous()
                expert_outputs += [expert(chunk)]
            expert_output = paddle.stack(expert_outputs, axis=1)  # [ecm]
            return expert_output

        assert self.gate is not None
        if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
            orig_shape_2 = input.shape
            input = input.reshape([self.world_size, self.num_local_experts, -1, input.shape[-1]])
            output = fwdfn(input)
            output += self.gate.weight.sum() * 0.0  # hack for grad
            output = output.reshape(orig_shape or orig_shape_2)  # [e*1,c,m]
            return output, None, 0

        capacity, dispatch_mask, combine_weights, scatter_index, router_loss = self.gate(input)
        self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])

        if self.world_size > 1:
            dispatched_input = _AllToAll.apply(dispatched_input, self.group)
        dispatched_input = dispatched_input.reshape([self.world_size * self.num_local_experts, capacity, d_model])
        expert_output = (
            recompute(fwdfn, dispatched_input) if self.recompute and self.training else fwdfn(dispatched_input)
        )
        d_model_out = expert_output.shape[-1]

        if self.world_size > 1:
            expert_output = _AllToAll.apply(expert_output, self.group)  # 拿到不同device上的expert计算结果

        expert_output = expert_output.reshape(
            [self.world_size * self.num_local_experts * capacity, d_model_out]
        )  # [e * 1, c, m]
        combined_output = combining(expert_output, combine_weights, scatter_index)

        if orig_shape:
            combined_output = combined_output.reshape(
                orig_shape[:-1]
                + [
                    d_model_out,
                ]
            )
        return combined_output, combine_weights, router_loss
