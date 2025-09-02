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
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    _check_environment_for_overlap,
)
from paddle.framework import core

from paddlenlp.transformers.llama.modeling_auto import get_mesh


def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm() or paddle.is_compiled_with_xpu():
        return hasattr(core.eager.ops.legacy, "fused_gemm_epilogue")
    else:
        return False


ipp = None
id2ipp = {}

paddle_nn_functional_linear = paddle.nn.functional.linear
if is_fused_matmul_bias_supported():
    paddle_incubate_nn_functional_fused_linear = paddle.incubate.nn.functional.fused_linear


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


if is_fused_matmul_bias_supported():
    origin_linear = paddle.incubate.nn.functional.fused_linear
else:
    origin_linear = paddle.nn.functional.linear


class FusedLinearWithReduceScatter(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None):
        global ipp
        input_parallel = dist.reshard(
            x,
            get_mesh(ipp),
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
            place = paddle.framework._current_expected_place()
            place = paddle.framework._get_paddle_place(place)

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
            place = paddle.framework._current_expected_place()
            place = paddle.framework._get_paddle_place(place)

            dx = paddle.Tensor(
                dx_local,
                dims=dx_global_shape,
                process_mesh=dinput_parallel.process_mesh,
                placements=[dist.Shard(1), dist.Shard(0)],
                place=place,
            )
            dx.stop_gradient = dx.stop_gradient
            return dx, dw, dbias


def forward_pre_hook(layer, input):
    paddle.nn.functional.linear = FusedLinearWithReduceScatter.apply
    if is_fused_matmul_bias_supported():
        paddle.incubate.nn.functional.fused_linear = FusedLinearWithReduceScatter.apply
    global ipp, id2ipp
    ipp = id2ipp[id(layer)]


def forward_post_hook(layer, input, output):
    paddle.nn.functional.linear = paddle_nn_functional_linear
    if is_fused_matmul_bias_supported():
        paddle.incubate.nn.functional.fused_linear = paddle_incubate_nn_functional_fused_linear


def mock_layers_sp_async_reduce_scatter(model):
    global ipp, id2ipp
    for name, layer in model.named_sublayers():
        if name.endswith("self_attn") or name.startswith("mlp"):
            ipp = layer.ipp
        for n in ["qkv_proj", "q_proj", "k_proj", "v_proj", "gate_up_fused_proj", "gate_proj", "up_proj"]:
            if name.endswith(n):
                id2ipp[id(layer)] = ipp
                layer.register_forward_pre_hook(forward_pre_hook)
                layer.register_forward_post_hook(forward_post_hook)
