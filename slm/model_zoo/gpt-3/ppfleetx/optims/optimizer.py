# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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

import inspect
import warnings

import paddle
from paddle import _C_ops
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
    device_guard,
)
from paddle.base import core, framework
from paddle.base.framework import Variable
from paddle.optimizer import Adam, AdamW, Momentum
from ppfleetx.distributed.apis import env
from ppfleetx.utils.tensor_fusion_helper import fused_parameters

__all__ = [
    "Adam",
    "AdamW",
    "Momentum",
    "FusedAdamW",
    "FusedOffloadAdamW",
]


class FusedAdamW(paddle.optimizer.AdamW):
    def __init__(self, learning_rate, parameters, grad_clip, **config):
        tensor_fusion = config.pop("tensor_fusion", False)

        if paddle.distributed.get_world_size() > 1:
            hcg = env.get_hcg()
            sharding_size = hcg.get_sharding_parallel_world_size()

        if tensor_fusion:
            self.decay_fused_tensors, self.all_fused_tensors = fused_parameters(parameters, sharding_size > 1)
            decay_params = [p.name for p in self.decay_fused_tensors]
        else:
            decay_params = [p.name for p in parameters if not any(nd in p.name for nd in ["bias", "norm", "b_0"])]

        apply_decay_param_fun = lambda x: x in decay_params

        super().__init__(
            learning_rate=learning_rate,
            parameters=self.all_fused_tensors if tensor_fusion else parameters,
            grad_clip=grad_clip,
            apply_decay_param_fun=apply_decay_param_fun,
            **config,
        )


class FusedOffloadAdamW(paddle.optimizer.AdamW):
    def __init__(self, learning_rate, parameters, grad_clip, **config):
        tensor_fusion = config.pop("tensor_fusion", False)

        if paddle.distributed.get_world_size() > 1:
            hcg = env.get_hcg()
            sharding_size = hcg.get_sharding_parallel_world_size()

        if tensor_fusion:
            self.decay_fused_tensors, self.all_fused_tensors = fused_parameters(parameters, sharding_size > 1)
            decay_params = [p.name for p in self.decay_fused_tensors]
        else:
            decay_params = [p.name for p in parameters if not any(nd in p.name for nd in ["bias", "norm", "b_0"])]

        apply_decay_param_fun = lambda x: x in decay_params

        super().__init__(
            learning_rate=learning_rate,
            parameters=self.all_fused_tensors if tensor_fusion else parameters,
            grad_clip=grad_clip,
            apply_decay_param_fun=apply_decay_param_fun,
            **config,
        )

        self._already_create_accumulater = set()
        self._dev_id = 0 if paddle.get_device() == "cpu" else int(paddle.get_device().split(":")[1])

        for p in parameters:
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                self._master_weights[p.name] = core.eager.Tensor(
                    name=p.name + "_fp32_master",
                    value=p.numpy(),
                    place=core.CPUPlace(),
                    stop_gradient=True,
                ).cast(paddle.float32)

        # support amsgrad: https://github.com/PaddlePaddle/Paddle/pull/68079
        self.support_amsgrad = (
            "amsgrad" in inspect.signature(paddle.optimizer.AdamW.__init__).parameters
        )

    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype):
            acc_dtype = paddle.float32
        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype, device="cpu")
        self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype, device="cpu")
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.9 if isinstance(self._beta1, Variable) else self._beta1,
            shape=[1],
            type=core.VarDesc.VarType.DENSE_TENSOR,
            device="cpu",
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.999 if isinstance(self._beta2, Variable) else self._beta2,
            shape=[1],
            type=core.VarDesc.VarType.DENSE_TENSOR,
            device="cpu",
        )

    def _create_accumulators(self, block, parameters):
        with device_guard():
            assert isinstance(block, framework.Block)
            if isinstance(parameters, dict):
                parameters = self._update_param_group(parameters)

            # Create accumulator tensors for first and second moments
            for p in parameters:
                if p.name in self._already_create_accumulater:
                    continue
                if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                    master_p = self._create_master_weight(p)
                    self._add_moments_pows(master_p)
                    self._already_create_accumulater.add(p.name)
                    continue
                if self._is_dtype_fp16_or_bf16(p.dtype) and not self._multi_precision:
                    warnings.warn(
                        "Accumulating with FP16 or BF16 in optimizer can lead to poor accuracy or slow convergence."
                        "Consider using multi_precision=True option of the Adam optimizer."
                    )
                self._add_moments_pows(p)
                self._already_create_accumulater.add(p.name)

    def _get_accumulator_master(self, name, param):
        """Utility function to fetch an accumulator for a parameter
        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched
        Returns:
            accumulator variable for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param.dtype)
        target_param = self._master_weights[param.name] if find_master else param
        target_name = target_param.name
        if name not in self._accumulators or target_name not in self._accumulators[name]:
            raise Exception("Accumulator {} does not exist for parameter {}".format(name, target_name))
        return self._accumulators[name][target_name]

    def _append_optimize_op(self, block, param_and_grad):
        with device_guard():
            assert isinstance(block, framework.Block)
            if isinstance(param_and_grad, dict):
                param_and_grad = self._update_param_group(param_and_grad)
            param, grad = param_and_grad

            # Whether we should do weight decay for the parameter.
            with_decay = True
            if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
                with_decay = False

            moment1 = self._get_accumulator_master(self._moment1_acc_str, param_and_grad[0])
            moment2 = self._get_accumulator_master(self._moment2_acc_str, param_and_grad[0])
            if self.support_amsgrad:
                moment2_max = (
                    self._get_accumulator_master(
                        self._moment2_acc_max_str, param_and_grad[0]
                    )
                    if self._amsgrad
                    else None
                )

            beta1_pow_acc = self._get_accumulator_master(self._beta1_pow_acc_str, param_and_grad[0])
            beta2_pow_acc = self._get_accumulator_master(self._beta2_pow_acc_str, param_and_grad[0])
            find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
            master_weight = self._master_weights[param_and_grad[0].name] if find_master else None
            lr = self._create_param_lr(param_and_grad)

            # create the adamw optimize op
            if framework.in_dygraph_mode():
                lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

                _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
                _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)

                origin_dtype = param_and_grad[0].dtype
                cpu_fp32_param = param_and_grad[0].cpu().cast(paddle.float32)
                cpu_fp32_grad = param_and_grad[1].cpu().cast(paddle.float32)

                if self.support_amsgrad:
                    _, _, _, _, _, _, _ = _C_ops.adamw_(
                        cpu_fp32_param,
                        cpu_fp32_grad,
                        lr.cpu(),
                        moment1.cpu(),
                        moment2.cpu(),
                        moment2_max, # moment2_max
                        beta1_pow_acc.cpu(),
                        beta2_pow_acc.cpu(),
                        master_weight.cpu() if master_weight is not None else None,
                        None,
                        _beta1,
                        _beta2,
                        self._epsilon,
                        lr_ratio_,
                        self._weight_decay,
                        with_decay,
                        self._lazy_mode,
                        1000,
                        find_master,
                        False,
                        self._amsgrad, # amsgrad
                    )
                else:
                    _, _, _, _, _, _ = _C_ops.adamw_(
                        cpu_fp32_param,
                        cpu_fp32_grad,
                        lr.cpu(),
                        moment1.cpu(),
                        moment2.cpu(),
                        beta1_pow_acc.cpu(),
                        beta2_pow_acc.cpu(),
                        master_weight.cpu() if master_weight is not None else None,
                        None,
                        _beta1,
                        _beta2,
                        self._epsilon,
                        lr_ratio_,
                        self._weight_decay,
                        with_decay,
                        self._lazy_mode,
                        1000,
                        find_master,
                        False,
                    )

                param_and_grad[0]._clear_data()
                cpu_fp32_param.cuda(self._dev_id).cast(origin_dtype)._share_buffer_to(param_and_grad[0])

                return None
