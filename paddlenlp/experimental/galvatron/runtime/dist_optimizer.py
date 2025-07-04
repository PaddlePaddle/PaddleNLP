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
from __future__ import annotations

import copy
import logging
import os
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import _C_ops, nn, pir
from paddle.amp.grad_scaler import OptimizerState
from paddle.autograd import PyLayer
from paddle.base import unique_name
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.base.framework import (
    EagerParamBase,
    Variable,
    default_main_program,
    in_dygraph_mode,
    in_pir_mode,
    use_pir_api,
)
from paddle.distributed import fleet
from paddle.distributed.auto_parallel import Engine, strategy as auto_strategy
from paddle.distributed.auto_parallel.interface import (
    shard_tensor as shard_tensor_static,
)
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.static.completion import (
    mark_as_sharding_propagation_skip_op,
)
from paddle.distributed.auto_parallel.static.dist_context import (
    get_default_distributed_context,
)
from paddle.distributed.auto_parallel.static.dist_op import DistributedOperator
from paddle.distributed.auto_parallel.static.utils import (
    convert_to_dims_mapping,
    fuse_param_func,
    get_dist_attr,
    split_mesh,
    split_param_func,
    to_list,
)
from paddle.framework import core
from paddle.io.dataloader.batch_sampler import (
    DistributedBatchSampler,
    _InfiniteIterableSampler,
)
from paddle.optimizer import Optimizer

# 以下相对路径的import需要修改
from .moe_utils import (
    _cal_local_shape,
    _dist_reshape,
    _NdMeshAlltoAll,
    _reshard_mesh_shape,
    _specific_alltoall_dim,
)
from .placement_type import (
    check_placements_equal,
    get_shard_spec,
    to_dim_map,
    to_placements,
)
from .random import determinate_rng, rng_state
from .sharding import ShardingOptimizerStage1, get_placement_with_sharding

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from typing_extensions import TypeAlias

    from paddle import Tensor
    from paddle._typing import (
        DTypeLike,
        NestedNumericSequence,
        PlaceLike,
        TensorLike,
    )
    from paddle.amp import GradScaler
    from paddle.base.framework import Program
    from paddle.distributed import Placement
    from paddle.distributed.auto_parallel.static.dist_input_spec import (
        DistributedInputSpec,
    )
    from paddle.io import DataLoader
    from paddle.metric import Metric
    from paddle.nn import Layer

    from .constants import (
        _AMPConfig,
        _DPOptimizationConfig,
        _FusedPassesConfig,
        _GradientMergeConfig,
        _MPOptimizationConfig,
        _PipelineConfig,
        _RecomputeConfig,
        _ShardingConfig,
        _SPOptimizationConfig,
    )

    _Mode: TypeAlias = Literal['train', 'eval', 'predict']

    class _Config(TypedDict, total=False):
        sharding: _ShardingConfig
        fused_passes: _FusedPassesConfig
        gradient_merge: _GradientMergeConfig
        pipeline: _PipelineConfig
        amp: _AMPConfig
        recompute: _RecomputeConfig
        mp_optimization: _MPOptimizationConfig
        dp_optimization: _DPOptimizationConfig
        sp_optimization: _SPOptimizationConfig


class _ShardOptimizer(Optimizer):
    def __init__(self, optimizer, shard_fn=None, gradient_accumulation_steps=1):
        assert (
            optimizer is not None
        ), "The argument `optimizer` cannot be empty."
        assert isinstance(
            optimizer, (paddle.optimizer.AdamW, paddle.optimizer.SGD)
        ), "`paddle.distributed.ShardOptimizer` only supports AdamW and SGD optimizer for now."

        # self.target_block = (
        #     paddle.base.framework.default_main_program().global_block()
        # )
        optimizer.helper = paddle.base.layer_helper.LayerHelper(
            optimizer.__class__.__name__
        )
        self.__dict__["_inner_opt"] = optimizer
        self._shard_clip = False
        if (
            hasattr(optimizer, "_grad_clip")
            and optimizer._grad_clip is not None
            and isinstance(optimizer._grad_clip, paddle.nn.ClipGradByGlobalNorm)
        ):
            self._shard_clip = True
        self._shard_fn = shard_fn
        self._sharding_axis = None
        self._sharding_degree = None
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if isinstance(
            self._shard_fn, (ShardingStage1, ShardingStage2, ShardingStage3)
        ):
            self._set_and_check_sharding_prop_from_param()
            self._shard_fn._set_sharding_axis(self._sharding_axis)

        # Invoke register hook for sharding stage 2 strategy
        if isinstance(self._shard_fn, ShardingStage2):
            for param in self._inner_opt._parameter_list:
                self._shard_fn._register_hook_for_param_grad(param)

        # Invoke shard_parameter in sharding stage 3 strategy
        if isinstance(self._shard_fn, ShardingStage3):
            for param in self._inner_opt._parameter_list:
                self._shard_fn._shard_parameter(param)

    def _set_and_check_sharding_prop_from_param(self):
        global_mesh = fleet.auto.get_mesh()
        if global_mesh:
            self._sharding_degree = global_mesh.get_dim_size(
                self._shard_fn._sharding_mesh_dim
            )
        elif self._shard_fn._mesh:
            self._sharding_degree = self._shard_fn._mesh.get_dim_size(
                self._shard_fn._sharding_mesh_dim
            )
        else:
            raise ValueError(
                "The global mesh or shard_fn mesh should be set for the sharding strategy."
            )

        # Note(luchang): Now we suggest using 0 axis as sharding axis.
        self._sharding_axis = 0

        # check the placement on sharding axis is Replicate
        param_list = self._inner_opt._parameter_list
        for param in param_list:
            if not param.is_dist():
                continue
            mesh = param.process_mesh
            placements = param.placements

            if not isinstance(placements[self._sharding_axis], dist.Replicate):
                # try to infer the sharding axis
                for dim, placement in enumerate(placements):
                    if isinstance(placement, dist.Replicate):
                        self._sharding_axis = dim

            # check the placement on sharding axis is Replicate
            assert isinstance(
                placements[self._sharding_axis], dist.Replicate
            ), "The placement on sharding_axis should be Replicate"

            # check the sharding degree since it has already been set,
            # skip check when mesh is true subset of global_mesh
            if global_mesh:
                if set(mesh.process_ids) < set(global_mesh.process_ids):
                    continue
            elif self._shard_fn._mesh:
                if set(mesh.process_ids) < set(
                    self._shard_fn._mesh.process_ids
                ):
                    continue
            else:
                assert (
                    mesh.dim_size(self._sharding_axis) == self._sharding_degree
                ), "The sharding degree of all parameters must be equal currently."

    def _shard_accumulator(self, param):
        target_name = param.name
        if param.name in self._inner_opt._master_weights.keys():
            master_weight = self._inner_opt._master_weights[param.name]
            target_name = master_weight.name
            # shard the master weight
            if self._shard_fn is not None:
                self._inner_opt._master_weights[param.name] = (
                    self._shard_fn.shard_master_weight(param, master_weight)
                )
                self._inner_opt._master_weights[param.name].name = target_name

        # shard the accumulators
        for key in self._inner_opt._accumulators.keys():
            accumulator = self._inner_opt._accumulators[key][target_name]
            if accumulator.is_dist() and not isinstance(accumulator, pir.Value):
                continue

            if paddle.in_dynamic_mode():
                origin_accumulator_name = accumulator.name

            if self._shard_fn is not None:
                self._inner_opt._accumulators[key][target_name] = (
                    self._shard_fn(key, param, accumulator)
                )
            else:
                if param.is_dist():
                    if 'beta' not in key:
                        # If param is a dist tensor should keep the shard info
                        # for accumulators except beta.
                        placements = param.placements
                    else:
                        # The beta should be replicated cross param's mesh
                        placements = [
                            dist.Replicate()
                            for _ in range(len(param.process_mesh.shape))
                        ]
                    self._inner_opt._accumulators[key][target_name] = (
                        shard_tensor(
                            accumulator,
                            mesh=param.process_mesh,
                            placements=placements,
                        )
                    )
            if paddle.in_dynamic_mode():
                self._inner_opt._accumulators[key][
                    target_name
                ].name = origin_accumulator_name

    def _reset_placements(self, param):
        if param.is_dist() and isinstance(
            self._shard_fn, (ShardingStage1, ShardingStage2)
        ):
            # in pir mode, reshard pass will automatically handle inplace case, so no extra work is required here.
            if not isinstance(param, pir.Value):
                new_placement = param.placements
                new_placement[self._sharding_axis] = dist.Replicate()
                out_param = dist.reshard(
                    param, param.process_mesh, new_placement
                )
                param.get_tensor()._share_data_with(out_param.get_tensor())

    def _create_accumulators(self, block, parameters):
        if isinstance(parameters, dict):
            parameters = parameters.get('params')
        # NOTE(zhiqiu): we need to create and shard accumulators for parameters one by one,
        # to avoid OOM caused by replcated accumulators.
        for p in parameters:
            self._inner_opt._create_accumulators(block, [p])
            self._shard_accumulator(p)

    def _finish_update(self, block, parameters_and_grads):
        self._inner_opt._finish_update(block, parameters_and_grads)
        if isinstance(parameters_and_grads, list):
            for p, _ in parameters_and_grads:
                self._reset_placements(p)
        else:
            # reset the parameter and grad to right placements
            for p, _ in parameters_and_grads['params']:
                self._reset_placements(p)

    def apply_gradients(self, params_grads):
        new_params_grads = []
        if self._shard_fn is not None:
            for param, grad in params_grads:
                new_params_grads.append(
                    (param, self._shard_fn("grad", param, grad))
                )
            return Optimizer.apply_gradients(self, new_params_grads)
        return Optimizer.apply_gradients(self, params_grads)

    def state_dict(self):
        """
        Create and shard the optimizer states e.g., accumulators and master_weights before load_state_dict.
        If training has already started or the optimizer states are already created and sharded, do nothing.
        """
        state_dict = self._inner_opt.state_dict()
        # training has already started.
        param_list = []
        if isinstance(self._inner_opt._parameter_list[0], dict):
            for param_group in self._inner_opt._parameter_list:
                param_list += param_group["params"]
        else:
            param_list = self._inner_opt._parameter_list
        for param in param_list:
            if param.stop_gradient:
                continue
            if hasattr(param, "main_grad"):
                if param.main_grad is not None:
                    return state_dict
            else:
                if param.grad is not None:
                    return state_dict

        # TODO(pangengzheng): deal with master_weights and LR_Scheduler later
        # the optimizer states are already created and sharded
        if any(
            v.is_dist()
            for k, v in state_dict.items()
            if k not in ["master_weights", "LR_Scheduler"]
        ):
            return state_dict

        # create and shard the optimizer states
        # fake the parameter gradient and invoke step to implicitly create the optimizer states.
        if not isinstance(self._inner_opt._parameter_list[0], dict):
            for param in self._inner_opt._parameter_list:
                if param.stop_gradient:
                    continue
                if hasattr(param, "main_grad"):
                    if param.main_grad is not None:
                        raise ValueError(
                            f"gradient should be None, but is {param.main_grad}"
                        )
                    param.main_grad = paddle.zeros_like(
                        param, dtype=paddle.float32
                    )
                else:
                    if param.grad is not None:
                        raise ValueError(
                            f"gradient should be None, but is {param.grad}"
                        )
                    param.grad = paddle.zeros_like(param, dtype=param.dtype)
        else:
            for param_group in self._inner_opt._param_groups:
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if hasattr(param, "main_grad"):
                        if param.main_grad is not None:
                            raise ValueError(
                                f"gradient should be None, but is {param.main_grad}"
                            )
                        param.main_grad = paddle.zeros_like(
                            param, dtype=paddle.float32
                        )
                    else:
                        if param.grad is not None:
                            raise ValueError(
                                f"gradient should be None, but is {param.grad}"
                            )
                        param.grad = paddle.zeros_like(param, dtype=param.dtype)
        self.step()
        # clear the parameter gradient
        self._inner_opt.clear_grad(set_to_zero=False)

        return self._inner_opt.state_dict()

    def _append_optimize_op(self, block, param_and_grad):
        if (
            in_auto_parallel_align_mode()  # In align mode, we use enable_delay_scale_loss by default
            and param_and_grad[1].is_dist()
        ):
            placements = param_and_grad[1].placements
            meshs = param_and_grad[1].process_mesh  # 通过这种方式去获取参数的mesh
            grad = param_and_grad[1]
            grad_mesh = grad.process_mesh

            def get_mesh(pp_idx=0):
                """
                获得pp_idx的mesh
                """
                mesh = fleet.auto.get_mesh()
                if "pp" in mesh.dim_names:
                    mesh = mesh.get_mesh_with_dim("pp", pp_idx)
                return mesh

            ipp = 0
            global_mesh = fleet.auto.get_mesh()
            if "pp" in global_mesh.dim_names:
                pp_degree = global_mesh.get_dim_size("pp")
                for i in range(pp_degree):
                    if meshs.process_ids == get_mesh(i).process_ids:
                        ipp = i
                        break

            change_mesh = False # 不对 在优化器步骤 只需要利用本层的梯度是完全参数更新即可
            if any(
                isinstance(placement, dist.Partial) for placement in placements
            ) and (
                (meshs.process_ids == get_mesh(ipp).process_ids)
                and (meshs.dim_names != get_mesh(ipp).dim_names)
            ):
                change_mesh = True

            if change_mesh:
                grad = dist.auto_parallel.moe_utils._dist_reshape(
                    grad,
                    grad.shape,
                    get_mesh(ipp),
                    [
                        dist.Partial(dist.ReduceType.kRedSum),
                        dist.Partial(dist.ReduceType.kRedSum),
                    ],
                )
                placements = grad.placements

            for i in range(len(placements) - 1, -1, -1):
                if isinstance(placements[i], dist.Partial):
                    placements[i] = dist.Replicate()
                    grad = dist.reshard(grad, grad.process_mesh, placements)
            if self.gradient_accumulation_steps > 1 and in_dygraph_mode():
                grad /= self.gradient_accumulation_steps

            if change_mesh:
                grad = dist.auto_parallel.moe_utils._dist_reshape(
                    grad, grad.shape, grad_mesh, [dist.Replicate()]
                )
            param_and_grad = (param_and_grad[0], grad)
        return self._inner_opt._append_optimize_op(block, param_and_grad)

    def __getattr__(self, item):
        if "_inner_opt" in self.__dict__:
            if item == "_inner_opt":
                return self.__dict__[item]
            return getattr(self.__dict__["_inner_opt"], item)
        else:
            raise AttributeError

    def __setattr__(self, item, value):
        if item == '_inner_opt':
            msg = f'{type(self).__name__}._inner_opt is READ ONLY'
            raise AttributeError(msg)
        return setattr(self._inner_opt, item, value)


class _ShardingStageBase:
    def __init__(self, mesh, sharding_mesh_dim):
        self._mesh = mesh
        self._sharding_axis = 0
        self._sharding_mesh_dim = sharding_mesh_dim

    def _set_sharding_axis(self, sharding_axis):
        self._sharding_axis = sharding_axis

    def shard_master_weight(
        self, param: Tensor, master_weight: Tensor
    ) -> Tensor:
        if param.is_dist():
            placements = get_placement_with_sharding(param, self._sharding_axis)
            if isinstance(master_weight, pir.Value):
                data_op = master_weight.get_defining_op()
                assert (
                    data_op.name() == "pd_op.data"
                ), "The master weight must be a result of data op."
                dim_map, partial_status = to_dim_map(
                    placements, len(master_weight.shape)
                )
                dist_attr = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        param.process_mesh, dim_map, partial_status
                    )
                )
                dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    master_weight.type(), dist_attr
                )
                master_weight.set_type(dist_type)
                data_op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        param.process_mesh, [], [dist_attr]
                    )
                )

            if paddle.in_dynamic_mode() and master_weight.is_dist():
                master_weight = reshard(
                    master_weight,
                    mesh=param.process_mesh,
                    placements=placements,
                )
        return master_weight


class ShardingStage1(_ShardingStageBase):
    """
    A builtin shard_fn for shard_optimizer interface, users can pass it to shard_optimizer to implement sharding optimization with stage 1.

    Args:
        sharding_mesh_dim(int|str): The sharding dimension in the mesh.
        mesh(None|paddle.distributed.ProcessMesh): If mesh is not None, the `ProcessMesh` object describes the Cartesian topology of the used processes for dense type parameters. Note: Currently, only one mesh configuration is supported for all dense parameters. If there is a need for multiple mesh configurations, please configure them yourself in the upper layer networking code.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> batch = paddle.rand(shape=[8, 8])
            >>> opt = paddle.optimizer.AdamW(parameters=layer.parameters())
            >>> opt = dist.shard_optimizer(opt, dist.ShardingStage1("x", mesh))
            >>> for _ in range(5):
            >>>     loss = layer(batch)
            >>>     loss.backward()
            >>>     opt.step()
            >>>     opt.clear_grad()
            >>> # This case need to be executed in multi-card environment
            >>> # python -m paddle.distributed.launch --gpus=0,1 {test_case}.py
    """

    def __init__(
        self,
        sharding_mesh_dim: int | str,
        mesh: ProcessMesh | None = None,
    ) -> None:
        super().__init__(mesh, sharding_mesh_dim)

    def __call__(self, key: str, param: Tensor, accumulator: Tensor) -> Tensor:
        if param.is_dist():
            # Only deal with momentum in optimizer, beta should be replicated cross param's mesh
            if 'beta' not in key:
                placements = get_placement_with_sharding(
                    param, self._sharding_axis
                )
            else:
                placements = [
                    dist.Replicate()
                    for _ in range(len(param.process_mesh.shape))
                ]

            if accumulator.is_dist():
                if accumulator.get_defining_op().name() == "pd_op.data":
                    dim_map, partial_status = (
                        dist.auto_parallel.placement_type.to_dim_map(
                            placements, len(accumulator.shape)
                        )
                    )
                    dist_attr = (
                        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            param.process_mesh, dim_map, partial_status
                        )
                    )
                    dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                        accumulator.type(), dist_attr
                    )
                    accumulator.set_type(dist_type)
                    op_dist_attr = (
                        paddle.base.libpaddle.pir.create_op_dist_attribute(
                            param.process_mesh, [], [dist_attr]
                        )
                    )
                    accumulator.get_defining_op().dist_attr = op_dist_attr
                    return accumulator
                return dist.reshard(accumulator, param.process_mesh, placements)
            else:
                return shard_tensor(
                    accumulator,
                    mesh=param.process_mesh,
                    placements=placements,
                )
        return accumulator


class ShardingStage2(_ShardingStageBase):
    """
    A builtin shard_fn for shard_optimizer interface, users can pass it to shard_optimizer to implement sharding optimization with stage 2.

    Args:
        sharding_mesh_dim(int|str): The sharding dimension name in the mesh.
        mesh(None|paddle.distributed.ProcessMesh): If mesh is not None, the `ProcessMesh` object describes the Cartesian topology of the used processes for dense type parameters. Note: Currently, only one mesh configuration is supported for all dense parameters. If there is a need for multiple mesh configurations, please configure them yourself in the upper layer networking code.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> batch = paddle.rand(shape=[8, 8])
            >>> opt = paddle.optimizer.AdamW(parameters=layer.parameters())
            >>> opt = dist.shard_optimizer(opt, dist.ShardingStage2("x", mesh))
            >>> for _ in range(5):
            >>>     loss = layer(batch)
            >>>     loss.backward()
            >>>     opt.step()
            >>>     opt.clear_grad()
            >>> # This case need to be executed in multi-card environment
            >>> # python -m paddle.distributed.launch --gpus=0,1 {test_case}.py
    """

    def __init__(
        self,
        sharding_mesh_dim: int | str,
        mesh: ProcessMesh | None = None,
    ) -> None:
        super().__init__(mesh, sharding_mesh_dim)

    def __call__(self, key: str, param: Tensor, accumulator: Tensor) -> Tensor:
        if param.is_dist():
            # Only deal with momentum in optimizer, beta should be replicated cross param's mesh
            if 'beta' not in key:
                placements = get_placement_with_sharding(
                    param, self._sharding_axis
                )
            else:
                placements = [
                    dist.Replicate()
                    for _ in range(len(param.process_mesh.shape))
                ]
            if accumulator.is_dist():
                if accumulator.get_defining_op().name() == "pd_op.data":
                    dim_map, partial_status = (
                        dist.auto_parallel.placement_type.to_dim_map(
                            placements, len(accumulator.shape)
                        )
                    )
                    dist_attr = (
                        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            param.process_mesh, dim_map, partial_status
                        )
                    )
                    dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                        accumulator.type(), dist_attr
                    )
                    accumulator.set_type(dist_type)
                    op_dist_attr = (
                        paddle.base.libpaddle.pir.create_op_dist_attribute(
                            param.process_mesh, [], [dist_attr]
                        )
                    )
                    accumulator.get_defining_op().dist_attr = op_dist_attr
                    return accumulator
                return dist.reshard(accumulator, param.process_mesh, placements)
            else:
                return shard_tensor(
                    accumulator,
                    mesh=param.process_mesh,
                    placements=placements,
                )
        return accumulator

    @staticmethod
    def _grad_hook(grad):
        # do reshard only if the grad is dist tensor and in partial status
        if grad.is_dist():
            partial_mesh_axis = None
            for mesh_axis, placement in enumerate(grad.placements):
                if isinstance(placement, dist.Partial):
                    partial_mesh_axis = mesh_axis
            if partial_mesh_axis is not None:
                new_placements = get_placement_with_sharding(
                    grad, partial_mesh_axis
                )
                return reshard(grad, grad.process_mesh, new_placements)

        return grad

    def _register_hook_for_param_grad(self, param):
        if param.is_dense() and self._mesh is not None:
            placements = []
            for _ in range(len(self._mesh.shape)):
                placements.append(dist.Replicate())
            param._to_dist_(placements, self._mesh)
        if param.is_dist():
            param.register_hook(ShardingStage2._grad_hook)


class ShardingStage3(_ShardingStageBase):
    """
    A builtin shard_fn for shard_optimizer interface, users can pass it to shard_optimizer to implement sharding optimization with stage 3.

    Args:
        sharding_mesh_dim(int|str): The sharding dimension name in the mesh.
        mesh(None|paddle.distributed.ProcessMesh): If mesh is not None, the `ProcessMesh` object describes the Cartesian topology of the used processes for dense type parameters. Note: Currently, only one mesh configuration is supported for all dense parameters. If there is a need for multiple mesh configurations, please configure them yourself in the upper layer networking code.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> batch = paddle.rand(shape=[8, 8])
            >>> opt = paddle.optimizer.AdamW(parameters=layer.parameters())
            >>> opt = dist.shard_optimizer(opt, dist.ShardingStage3("x", mesh))
            >>> for _ in range(5):
            >>>     loss = layer(batch)
            >>>     loss.backward()
            >>>     opt.step()
            >>>     opt.clear_grad()
            >>> # This case need to be executed in multi-card environment
            >>> # python -m paddle.distributed.launch --gpus=0,1 {test_case}.py
    """

    def __init__(
        self,
        sharding_mesh_dim: int | str,
        mesh: ProcessMesh | None = None,
    ) -> None:
        super().__init__(mesh, sharding_mesh_dim)

    def _shard_parameter(self, param):
        if param.is_dense() and self._mesh is not None:
            placements = []
            for _ in range(len(self._mesh.shape)):
                placements.append(dist.Replicate())
            param._to_dist_(placements, self._mesh)
        if param.is_dist():
            new_placements = get_placement_with_sharding(
                param, self._sharding_axis
            )
            shard_param = dist.reshard(  # 此处完成ZeRO-3的参数shard操作
                param, param.process_mesh, new_placements
            )
            # change the holder of param to new shard_param
            param.get_tensor()._share_data_with(shard_param.get_tensor())

    def _unshard_parameter(self, param):
        if param.is_dist():
            new_placements = param.placements
            if isinstance(new_placements[self._sharding_axis], dist.Shard):
                new_placements[self._sharding_axis] = dist.Replicate()

            new_param = dist.reshard(param, param.process_mesh, new_placements)
            param.get_tensor()._share_data_with(new_param.get_tensor())

    def __call__(self, key: str, param: Tensor, accumulator: Tensor) -> Tensor:
        if param.is_dist():
            # Only deal with momentum in optimizer, beta should be replicated cross param's mesh
            if 'beta' not in key:
                placements = param.placements
                if all(
                    isinstance(placement, dist.Replicate)
                    for placement in placements
                ):
                    placements = get_placement_with_sharding(
                        param, self._sharding_axis
                    )

            else:
                placements = [
                    dist.Replicate()
                    for _ in range(len(param.process_mesh.shape))
                ]
            return shard_tensor(
                accumulator,
                mesh=param.process_mesh,
                placements=placements,
            )
        return accumulator


def shard_optimizer(
    optimizer: Optimizer,
    shard_fn: Callable[[str, Tensor, Tensor], Tensor] | None = None,
    gradient_accumulation_steps: int = 1,
) -> _ShardOptimizer:
    return _ShardOptimizer(optimizer, shard_fn, gradient_accumulation_steps)