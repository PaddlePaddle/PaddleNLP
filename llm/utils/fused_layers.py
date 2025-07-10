# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os

import paddle
import paddle.distributed as dist
from paddle import _C_ops
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.log_util import logger
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    _check_environment_for_overlap,
)
from paddle.framework import core

_raise_cuda_env_unset_warning = True
_mp_async_allreduce = False
_sp_async_reduce_scatter = False
paddle_nn_functional_linear = paddle.nn.functional.linear


def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm() or paddle.is_compiled_with_xpu():
        return hasattr(core.eager.ops.legacy, "fused_gemm_epilogue")
    else:
        return False


if is_fused_matmul_bias_supported():
    origin_linear = paddle.incubate.nn.functional.fused_linear
    paddle_incubate_nn_functional_fused_linear = paddle.incubate.nn.functional.fused_linear
else:
    origin_linear = paddle.nn.functional.linear


def mp_async_allreduce(x_grad):
    if _mp_async_allreduce and x_grad.process_mesh is not None:
        # Using small operation to preempt GPU SMs for all_reduce to achieve overlap.
        if int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "0")) != 1:
            global _raise_cuda_env_unset_warning
            if _raise_cuda_env_unset_warning:
                logger.warning(
                    "You set mp_async_allreduce=True, but you forget to set environment "
                    "variable CUDA_DEVICE_MAX_CONNECTIONS=1, which may leads to performance "
                    "loss. Try to export CUDA_DEVICE_MAX_CONNECTIONS=1 for better performance."
                )
                _raise_cuda_env_unset_warning = False

        mp_placement_index = x_grad.process_mesh.dim_names.index("mp")
        if mp_placement_index != -1 and x_grad.placements[mp_placement_index].is_partial():
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            task = dist.stream.all_reduce(
                x_grad._local_value(),
                group=model_parallel_group,
                sync_op=False,
            )
            return task
        else:
            return None
    else:
        return None


def sync_mp_allreduce(task, dist_tensor):
    mp_placement_index = dist_tensor.process_mesh.dim_names.index("mp")
    new_placments = list()
    for idx, placment in enumerate(dist_tensor.placements):
        if idx == mp_placement_index:
            new_placments.append(dist.Replicate())
        else:
            new_placments.append(placment)
    place = paddle.framework._current_expected_place()
    place = paddle.framework._get_paddle_place(place)

    task.wait()

    return paddle.Tensor(
        dist_tensor._local_value(),
        dims=dist_tensor.shape,
        process_mesh=dist_tensor.process_mesh,
        placements=new_placments,
        place=place,
    )


# modify from Paddle/python/paddle/distributed/auto_parallel/moe_utils.py
def _dist_reshape(
    dist_tensor,
    global_shape,
    mesh,
    placements,
):
    local_tensor = dist_tensor._local_value()
    tgt_global_shape = [dist_tensor.shape[0] * dist_tensor.shape[1], dist_tensor.shape[2]]
    tgt_local_shape = [local_tensor.shape[0] * local_tensor.shape[1], local_tensor.shape[2]]

    place = paddle.framework._current_expected_place()
    place = paddle.framework._get_paddle_place(place)

    local_tensor = local_tensor.reshape(tgt_local_shape)

    if placements[1].is_shard():
        new_placements = [dist.Shard(0), dist.Shard(1)]
    else:
        new_placements = [dist.Shard(0), dist.Replicate()]

    out = paddle.Tensor(
        local_tensor,
        dims=tgt_global_shape,
        process_mesh=mesh,
        placements=new_placements,
        place=place,
    )
    out.stop_gradient = dist_tensor.stop_gradient
    return out


class FusedLinearWithGradAdd(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None):
        y = origin_linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, y_grad):
        x, weight, bias = ctx.saved_tensor()
        x_grad = paddle.matmul(y_grad, weight, transpose_y=True)

        mp_task = mp_async_allreduce(x_grad)

        # _C_ops.fused_linear_param_grad_add(x, y_grad, dw, db, multi precision, has bias)
        if bias is None:
            if hasattr(weight, "main_grad"):
                weight.main_grad, _ = _C_ops.fused_linear_param_grad_add(
                    x, y_grad, weight.main_grad, None, True, False
                )
                if mp_task is not None:
                    x_grad = sync_mp_allreduce(mp_task, x_grad)
                return x_grad, None
            else:
                if weight.grad is not None:
                    weight.grad, _ = _C_ops.fused_linear_param_grad_add(
                        x, y_grad, weight.grad, None, False if weight.grad.dtype != paddle.float32 else True, False
                    )
                    if mp_task is not None:
                        x_grad = sync_mp_allreduce(mp_task, x_grad)
                    return x_grad, None
                else:
                    weight_grad, _ = _C_ops.fused_linear_param_grad_add(x, y_grad, None, None, False, False)
                    if mp_task is not None:
                        x_grad = sync_mp_allreduce(mp_task, x_grad)
                    return x_grad, weight_grad

        if hasattr(weight, "main_grad") and hasattr(bias, "main_grad"):
            weight.main_grad, bias.main_grad = _C_ops.fused_linear_param_grad_add(
                x, y_grad, weight.main_grad, bias.main_grad, True, True
            )
            if mp_task is not None:
                x_grad = sync_mp_allreduce(mp_task, x_grad)
            return x_grad, None, None
        else:
            if weight.grad is not None:
                assert bias.grad is not None
                weight.grad, bias.grad = _C_ops.fused_linear_param_grad_add(
                    x, y_grad, weight.grad, bias.grad, False if weight.grad.dtype != paddle.float32 else True, True
                )
                if mp_task is not None:
                    x_grad = sync_mp_allreduce(mp_task, x_grad)
                return x_grad, None, None
            else:
                weight_grad, bias_grad = _C_ops.fused_linear_param_grad_add(x, y_grad, None, None, False, True)
                if mp_task is not None:
                    x_grad = sync_mp_allreduce(mp_task, x_grad)
                return x_grad, weight_grad, bias_grad


class OverlapLinear(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None):
        y = origin_linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, y_grad):
        x, weight, bias = ctx.saved_tensor()
        x_grad = paddle.matmul(y_grad, weight, transpose_y=True)

        mp_task = mp_async_allreduce(x_grad)

        y_grad = y_grad.reshape([-1, y_grad.shape[-1]])
        weight_grad = paddle.matmul(
            x.reshape([-1, x.shape[-1]]),
            y_grad,
            transpose_x=True,
        )
        if bias is None:
            if mp_task is not None:
                x_grad = sync_mp_allreduce(mp_task, x_grad)
            return x_grad, weight_grad
        else:
            bias_grad = paddle.sum(y_grad, axis=0)
            if mp_task is not None:
                x_grad = sync_mp_allreduce(mp_task, x_grad)
            return x_grad, weight_grad, bias_grad


class FusedLinearWithReduceScatter(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None):
        input_parallel = dist.reshard(
            x,
            x.process_mesh,
            [dist.Shard(1), dist.Replicate()],
        )
        y = origin_linear(input_parallel, weight, bias)
        ctx.save_for_backward(weight, bias, input_parallel)

        return y

    @staticmethod
    def backward(ctx, dy):
        weight, bias, input_parallel = ctx.saved_tensor()

        # compute dx
        if dy.dtype == weight.dtype:
            dinput_parallel = paddle.matmul(dy, weight, transpose_y=True)
        else:
            dinput_parallel = paddle.matmul(dy, paddle.cast(weight, dtype=dy.dtype), transpose_y=True)

        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        parallelism = model_parallel_group.nranks

        assert (
            dinput_parallel.shape[0] % parallelism == 0
        ), f"Input sequence length {dinput_parallel.shape[0]} can't be divided exactly by sequence parallelism {parallelism}"

        # reduce-scatter dx
        dx_global_shape = dinput_parallel.shape
        dx_global_shape[0] = dx_global_shape[0] // parallelism
        dinput_parallel_local = dinput_parallel._local_value()
        dx_local_shape = dinput_parallel_local.shape
        dx_local_shape[0] = dx_local_shape[0] // parallelism
        dx_local = paddle.empty(shape=dx_local_shape, dtype=dinput_parallel.dtype)
        task = dist.stream.reduce_scatter(
            dx_local,
            dinput_parallel_local,
            op=dist.ReduceOp.SUM,
            group=model_parallel_group,
            sync_op=False,
        )

        # compute dw and dbias
        _check_environment_for_overlap()

        place = paddle.framework._current_expected_place()
        place = paddle.framework._get_paddle_place(place)

        if _sp_async_reduce_scatter:
            if bias is None:
                if hasattr(weight, "main_grad"):
                    (weight.main_grad, _,) = paddle._C_ops.fused_linear_param_grad_add(
                        input_parallel, dy, weight.main_grad, None, True, False
                    )
                    task.wait()
                    dx = paddle.Tensor(
                        dx_local,
                        dims=dx_global_shape,
                        process_mesh=dinput_parallel.process_mesh,
                        placements=[dist.Shard(1), dist.Shard(0)],
                        place=place,
                    )
                    dx.stop_gradient = dx.stop_gradient
                    return dx, None
                else:
                    if weight.grad is not None:
                        (weight.grad, _,) = paddle._C_ops.fused_linear_param_grad_add(
                            input_parallel, dy, weight.grad, None, False, False
                        )
                        task.wait()
                        dx = paddle.Tensor(
                            dx_local,
                            dims=dx_global_shape,
                            process_mesh=dinput_parallel.process_mesh,
                            placements=[dist.Shard(1), dist.Shard(0)],
                            place=place,
                        )
                        dx.stop_gradient = dx.stop_gradient
                        return dx, None
                    else:
                        (
                            dw,
                            _,
                        ) = paddle._C_ops.fused_linear_param_grad_add(input_parallel, dy, None, None, False, False)
                        task.wait()
                        dx = paddle.Tensor(
                            dx_local,
                            dims=dx_global_shape,
                            process_mesh=dinput_parallel.process_mesh,
                            placements=[dist.Shard(1), dist.Shard(0)],
                            place=place,
                        )
                        dx.stop_gradient = dx.stop_gradient
                        return dx, dw

            if hasattr(weight, "main_grad") and hasattr(bias, "main_grad"):
                (weight.main_grad, bias.main_grad,) = paddle._C_ops.fused_linear_param_grad_add(
                    input_parallel,
                    dy,
                    weight.main_grad,
                    bias.main_grad,
                    True,
                    True,
                )
                task.wait()
                dx = paddle.Tensor(
                    dx_local,
                    dims=dx_global_shape,
                    process_mesh=dinput_parallel.process_mesh,
                    placements=[dist.Shard(1), dist.Shard(0)],
                    place=place,
                )
                dx.stop_gradient = dx.stop_gradient
                return dx, None, None
            else:
                if weight.grad is not None:
                    assert bias.grad is not None
                    (weight.grad, bias.grad,) = paddle._C_ops.fused_linear_param_grad_add(
                        input_parallel, dy, weight.grad, bias.grad, False, True
                    )
                    task.wait()
                    dx = paddle.Tensor(
                        dx_local,
                        dims=dx_global_shape,
                        process_mesh=dinput_parallel.process_mesh,
                        placements=[dist.Shard(1), dist.Shard(0)],
                        place=place,
                    )
                    dx.stop_gradient = dx.stop_gradient
                    return dx, None, None
                else:
                    # When main_grad is not enabled and gradient_accumulation is used, the grad is not initialized for the first acc step.
                    (
                        dw,
                        dbias,
                    ) = paddle._C_ops.fused_linear_param_grad_add(input_parallel, dy, None, None, False, True)
                    task.wait()
                    dx = paddle.Tensor(
                        dx_local,
                        dims=dx_global_shape,
                        process_mesh=dinput_parallel.process_mesh,
                        placements=[dist.Shard(1), dist.Shard(0)],
                        place=place,
                    )
                    dx.stop_gradient = dx.stop_gradient
                    return dx, dw, dbias
        else:
            dy = _dist_reshape(dy, [-1, dy.shape[-1]], dy.process_mesh, dy.placements)
            input_parallel = _dist_reshape(
                input_parallel, [-1, input_parallel.shape[-1]], input_parallel.process_mesh, input_parallel.placements
            )
            dw = paddle.matmul(
                input_parallel,
                dy,
                transpose_x=True,
            )
            if bias is None:
                task.wait()
                dx = paddle.Tensor(
                    dx_local,
                    dims=dx_global_shape,
                    process_mesh=dinput_parallel.process_mesh,
                    placements=[dist.Shard(1), dist.Shard(0)],
                    place=place,
                )
                dx.stop_gradient = dx.stop_gradient
                return dx, dw
            else:
                dbias = paddle.sum(dy, axis=0)
                task.wait()
                dx = paddle.Tensor(
                    dx_local,
                    dims=dx_global_shape,
                    process_mesh=dinput_parallel.process_mesh,
                    placements=[dist.Shard(1), dist.Shard(0)],
                    place=place,
                )
                dx.stop_gradient = dx.stop_gradient
                return dx, dw, dbias


def mock_layers(
    model=None,
    enable_fused_linear_grad_add=True,
    enable_mp_async_allreduce=False,
    enable_sp_async_reduce_scatter=False,
):
    global _mp_async_allreduce
    global _sp_async_reduce_scatter
    _mp_async_allreduce = enable_mp_async_allreduce
    _sp_async_reduce_scatter = enable_sp_async_reduce_scatter

    if enable_fused_linear_grad_add:
        paddle.nn.functional.linear = FusedLinearWithGradAdd.apply
        if is_fused_matmul_bias_supported():
            paddle.incubate.nn.functional.fused_linear = FusedLinearWithGradAdd.apply
    else:
        paddle.nn.functional.linear = OverlapLinear.apply
        if is_fused_matmul_bias_supported():
            paddle.incubate.nn.functional.fused_linear = OverlapLinear.apply

    if enable_sp_async_reduce_scatter:

        def forward_pre_hook(layer, input):
            paddle.nn.functional.linear = FusedLinearWithReduceScatter.apply
            if is_fused_matmul_bias_supported():
                paddle.incubate.nn.functional.fused_linear = FusedLinearWithReduceScatter.apply

        def forward_post_hook(layer, input, ouput):
            paddle.nn.functional.linear = paddle_nn_functional_linear
            if is_fused_matmul_bias_supported():
                paddle.incubate.nn.functional.fused_linear = paddle_incubate_nn_functional_fused_linear

        for name, layer in model.named_sublayers():
            for n in ["qkv_proj", "q_proj", "k_proj", "v_proj", "gate_up_fused_proj", "gate_proj", "up_proj"]:
                if name.endswith(n):
                    layer.register_forward_pre_hook(forward_pre_hook)
                    layer.register_forward_post_hook(forward_post_hook)
