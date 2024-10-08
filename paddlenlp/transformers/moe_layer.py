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
import paddle.nn as nn
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
        input: paddle.Tensor,
        group: Group,
    ) -> paddle.Tensor:
        """
        All-to-all communication in the group.

        Args:
            ctx (Any): Context object.
            input (paddle.Tensor): Input tensor.
            group (Group): The group object.

        Returns:
            Tensor: Output tensor.
        """
        ctx.group = group

        if group is not None and not group.is_member():
            # when process is not in the group, return input
            return input

        if dist.get_world_size(group) <= 1:
            # when world size is 1, return input
            return input

        output = paddle.empty_like(input)
        stream.alltoall_single(output, input, None, None, group, True, True)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: paddle.Tensor) -> Tuple[paddle.Tensor]:
        """
        Aggregates gradient information from all input tensors into a single tensor.

        Args:
            ctx (Any): The context object used to store information that needs to be passed.
            *grad_output (paddle.Tensor): A list of input tensors whose gradients are to be aggregated.

        Returns:
            Tuple[paddle.Tensor]: A tuple containing a tensor that holds the gradients of all input tensors.

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
    Returns:
        output
        combine_weight
        router-loss

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle
            >>> import paddle.nn as nn
            >>> import paddle.distributed as dist

            >>> class SimpleNet(nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self._linear = nn.Linear(10, 1)
            ...     def forward(self, x):
            ...         return self._linear(x)

            >>> dist.init_parallel_env()
            >>> model = SimpleNet()
            >>> dp_model = paddle.DataParallel(model)

            >>> inputs_1 = paddle.randn([10, 10], 'float32')
            >>> inputs_2 = paddle.ones([10, 10], 'float32')

            >>> with dp_model.no_sync():
            ...     # gradients will not be synchronized
            ...     dp_model(inputs_1).backward()

            >>> # synchronization happens here
            >>> dp_model(inputs_2).backward()
    """

    def __init__(
        self,
        gate: nn.Layer,
        num_experts: int,
        experts: List[nn.Layer],
        group: Group = None,
        all_to_all_dropout=0.0,
    ):
        super().__init__()
        self.gate = gate

        self.num_experts = num_experts
        self.experts = experts

        self.group = group
        self.all_to_all_dropout = all_to_all_dropout

        self.enable_recompute = False

        self.world_size = 1 if dist.get_world_size(self.group) < 1 else dist.get_world_size(group)
        is_dummy_moe = dist.get_world_size(group) == 1
        self.rank = 0 if dist.get_rank(self.group) < 0 else dist.get_rank(self.group)

        for p in self.gate.parameters():
            p.is_gate = True

        for k in experts:
            if k is not None:
                for p in k.parameters():
                    p.expert = not is_dummy_moe
                    p.no_sync = not is_dummy_moe

        assert (
            num_experts // self.world_size == 0
        ), f"num_experts must be divisible by world_size, got: {num_experts} vs {self.world_size}"
        self.num_local_experts = num_experts // self.world_size

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
