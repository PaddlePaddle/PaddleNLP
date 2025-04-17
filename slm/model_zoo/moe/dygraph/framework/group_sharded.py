# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from types import MethodType

import paddle

# New version
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import (
    GroupShardedOptimizerStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import (
    GroupShardedStage2,
)
from paddle.incubate.distributed.models.moe.grad_clip import ClipGradForMOEByGlobalNorm
from paddle.optimizer import Optimizer


class ClipGradForShardedMOEByGlobalNorm(ClipGradForMOEByGlobalNorm):
    @paddle.no_grad()
    def _dygraph_clip(self, params_grads):
        normal_params_grads = []
        moe_params_grads = []

        # separate moe params from normal params
        if self.moe_group is not None and self.moe_group.nranks > 1:
            for p, g in params_grads:
                if self.is_expert_param_func(p):
                    moe_params_grads.append((p, g))
                else:
                    normal_params_grads.append((p, g))
        else:
            normal_params_grads = params_grads

        # why to return sum_dtype?
        # we will call `get_l2_norm_pow` twice and the precisions may be different.
        # For convenience and simplification, we use sum_dtype directly instead of global_norm_var_normal.dtype
        global_norm_var_normal, sum_dtype = self.get_l2_norm_pow(normal_params_grads)
        if global_norm_var_normal is not None:
            paddle.distributed.all_reduce(
                global_norm_var_normal, op=paddle.distributed.ReduceOp.SUM, group=self.moe_group
            )

        global_norm_var_moe = None
        if len(moe_params_grads) > 0:
            global_norm_var_moe, _ = self.get_l2_norm_pow(moe_params_grads, sum_dtype)
            if global_norm_var_moe is not None:
                paddle.distributed.all_reduce(
                    global_norm_var_moe, op=paddle.distributed.ReduceOp.SUM, group=self.moe_group
                )

        if global_norm_var_normal is None and global_norm_var_moe is None:
            return params_grads
        elif global_norm_var_normal is None:
            global_norm_var = global_norm_var_moe
        elif global_norm_var_moe is None:
            global_norm_var = global_norm_var_normal
        else:
            if global_norm_var_normal.dtype != global_norm_var_moe.dtype:
                # compared with normal norm, moe norm is the later one,
                # so its precision is no lower than normal norm
                global_norm_var_normal = global_norm_var_normal.astype(global_norm_var_moe.dtype)
            global_norm_var = global_norm_var_normal + global_norm_var_moe

        params_and_grads = []
        global_norm_var = paddle.sqrt(global_norm_var)
        max_global_norm = paddle.full(shape=[1], dtype=global_norm_var.dtype, fill_value=self.clip_norm)
        clip_var = paddle.maximum(x=max_global_norm, y=paddle.maximum(x=global_norm_var, y=max_global_norm))
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, "need_clip", True) is False:
                params_and_grads.append((p, g))
                continue
            # TODO(wangxi): use inplace elementwise_mul
            clip_input = clip_var.astype("float16") if g.dtype == paddle.float16 else clip_var
            new_grad = paddle.multiply(x=g, y=clip_input)
            params_and_grads.append((p, new_grad))
        return params_and_grads


def group_sharded_parallel(
    model, optimizer, group=None, offload=False, sync_buffers=False, buffer_max_size=2**23, segment_size=2**20
):

    # check optition type
    assert isinstance(model, paddle.nn.Layer), "The model must be the instance of paddle.nn.Layer."
    assert isinstance(optimizer, Optimizer), "The optimizer must be the instance of paddle.optimizer.Optimizer."

    def check_dtype(param):
        return param.dtype == paddle.float16

    sharded_params = []
    pretreated_params = []
    for p in optimizer._parameter_list:
        if "expert" not in p.name and "gate" not in p.name:
            sharded_params.append(p)
        else:
            pretreated_params.append(p)

    opt_gc = optimizer._grad_clip
    if opt_gc is not None:
        optimizer._grad_clip = ClipGradForShardedMOEByGlobalNorm(
            opt_gc.clip_norm, opt_gc.is_expert_param_func, opt_gc.moe_group, opt_gc.group_name
        )

    # convert model/optimizer
    optimizer = GroupShardedOptimizerStage2(params=sharded_params, optim=optimizer, group=group, offload=offload)
    model = GroupShardedStage2(
        model, optimizer, group=group, sync_buffers=sync_buffers, buffer_max_size=buffer_max_size
    )

    clear_func = model._clear_gradients
    for opt in model._sharding_optimizers:

        def _opt_clear(self):
            clear_func()
            for p in pretreated_params:
                if p.grad is not None:
                    p.grad.zero_()

        opt.clear_grad = MethodType(_opt_clear, opt)

    return model, optimizer
