# !/usr/bin/env python3

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

""" mc2(tp overlap) """

import paddle
import paddle_custom_device
from paddle.autograd import PyLayer


class MC2LoRaRowParallelLinear(PyLayer):
    @staticmethod
    def forward(ctx, input_, weight, group):
        ctx.save_for_backward(input_, weight)
        rank = paddle.distributed.get_rank()
        hcom_name = group.process_group.get_comm_name(rank)
        x = input_.reshape([-1, input_.shape[-1]])
        out = paddle_custom_device.npu.fused_mm_allreduce(
            x, weight, bias=None, hcom=hcom_name, reduce_op="sum", comm_turn=0
        )
        output = out.reshape([input_.shape[0], input_.shape[1], weight.shape[1]])
        ctx.ring_id = group.id
        return output

    @staticmethod
    def backward(ctx, dy):
        input_, weight = ctx.saved_tensor()
        out_grad = dy
        sub_grad = out_grad.reshape([-1, out_grad.shape[-1]])
        input_grad = paddle.matmul(sub_grad, weight.t())
        if weight.stop_gradient:
            return input_grad.reshape(input_.shape)
        else:
            input_reshape = input_.reshape([-1, input_.shape[-1]])
            weight_grad = input_reshape.t().matmul(sub_grad)
            return input_grad.reshape(input_.shape), weight_grad


class MC2LoRaColumnParallelLinear(PyLayer):
    @staticmethod
    def forward(ctx, input_, weight, group):
        ctx.save_for_backward(input_, weight)
        ctx.group = group
        input_mp = input_
        result_mp = paddle.matmul(input_mp, weight)
        return result_mp

    @staticmethod
    def backward(ctx, dy):
        input_, weight = ctx.saved_tensor()
        sub_grad = dy.reshape([-1, dy.shape[-1]])
        rank = paddle.distributed.get_rank()
        hcom_name = ctx.group.process_group.get_comm_name(rank)

        d_weight = input_.reshape([-1, input_.shape[-1]]).t().matmul(sub_grad) if not weight.stop_gradient else None
        d_input = paddle_custom_device.npu.fused_mm_allreduce(
            sub_grad, weight.t(), bias=None, hcom=hcom_name, reduce_op="sum", comm_turn=0
        )

        if d_weight is not None:
            return d_input.reshape(input_.shape), d_weight
        else:
            return d_input.reshape(input_.shape)
