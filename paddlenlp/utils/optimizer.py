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

from collections import defaultdict

import paddle
import paddle.autograd as imperative_base
from paddle import pir
from paddle.base import core, framework
from paddle.base.framework import Variable, in_dynamic_or_pir_mode, in_pir_mode
from paddle.base.libpaddle import DataType
from paddle.optimizer.adamw import AdamW
from paddle.pir import Value


class AdamWMini(AdamW):
    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype):
            acc_dtype = DataType.FLOAT32 if in_pir_mode() else paddle.float32

        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        # change moment2
        self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype, shape=[1])
        try:
            type = core.VarDesc.VarType.DENSE_TENSOR
        except:
            type = core.VarDesc.VarType.LOD_TENSOR
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.9 if isinstance(self._beta1, (Variable, Value)) else self._beta1,
            shape=[1],
            type=type,
            device="cpu",
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.999 if isinstance(self._beta2, (Variable, Value)) else self._beta2,
            shape=[1],
            type=type,
            device="cpu",
        )

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param = param_and_grad[0]

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
            with_decay = False

        moment1 = self._get_accumulator_master(self._moment1_acc_str, param_and_grad[0])
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param_and_grad[0])
        beta1_pow_acc = self._get_accumulator_master(self._beta1_pow_acc_str, param_and_grad[0])
        beta2_pow_acc = self._get_accumulator_master(self._beta2_pow_acc_str, param_and_grad[0])
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
        master_weight = self._master_weights[param_and_grad[0].name] if find_master else None
        lr = self._create_param_lr(param_and_grad)
        # create the adamw optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)

            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None
            self.adamw_python(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                moment1,
                moment2,
                beta1_pow_acc,
                beta2_pow_acc,
                master_weight,
                found_inf,
                _beta1,
                _beta2,
                self._epsilon,
                lr_ratio_,
                self._weight_decay,
                with_decay,
                find_master,
            )
            return None
        else:
            raise NotImplementedError("Not implemented yet.")

    def adamw_python(
        self,
        param,
        grad,
        learning_rate,
        moment1,
        moment2,
        beta1_pow,
        beta2_pow,
        master_weight,
        skip_update,
        beta1,
        beta2,
        epsilon,
        lr_ratio,
        coeff,
        with_decay,
        multi_precision,
    ):
        if skip_update:
            return
        if not with_decay:
            coeff = 0.0
        if not multi_precision:
            master_weight = None
        lr = learning_rate * lr_ratio
        if master_weight is not None:
            p = master_weight
        else:
            p = param
        p *= 1.0 - lr * coeff
        mom1 = moment1
        mom2 = moment2

        mom1 = beta1 * mom1 + (1.0 - beta1) * grad
        mom2 = beta2 * mom2 + (1.0 - beta2) * (grad * grad).mean()
        denom = mom2.sqrt() / (1.0 - beta2_pow).sqrt() + epsilon
        p += (moment1 / denom) * (-(lr / (1.0 - beta1_pow)))
        if master_weight is not None:
            master_weight[:] = p
            param[:] = p.astype(param.dtype)
        else:
            param[:] = p
        moment1[:] = mom1
        moment2[:] = mom2
        beta1_pow[:], beta2_pow[:] = beta1 * beta1_pow[:], beta2 * beta2_pow[:]
        # 看看怎么更新
        return


class LoRAPro(AdamW):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        parameters=None,
        weight_decay: float = 0.01,
        lr_ratio=None,
        apply_decay_param_fun=None,
        grad_clip=None,
        lazy_mode: bool = False,
        multi_precision: bool = False,
        amsgrad: bool = False,
        name=None,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__(
            learning_rate,
            beta1,
            beta2,
            epsilon,
            parameters,
            weight_decay,
            lr_ratio,
            apply_decay_param_fun,
            grad_clip,
            lazy_mode,
            multi_precision,
            amsgrad,
            name,
        )
        self.scaling_factor = scaling_factor

    @imperative_base.no_grad()
    @framework.non_static_only
    def step(self) -> None:
        """
        Execute the optimizer and update parameters once.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> a = paddle.arange(26, dtype="float32").reshape([2, 13])
                >>> linear = paddle.nn.Linear(13, 5)
                >>> # This can be any optimizer supported by dygraph.
                >>> adam = paddle.optimizer.Adam(learning_rate = 0.01,
                ...                         parameters = linear.parameters())
                >>> out = linear(a)
                >>> out.backward()
                >>> adam.step()
                >>> adam.clear_grad()
        """
        if paddle.base.dygraph.base.in_to_static_mode():
            self._declarative_step()
            return
        scaling_factor = self.scaling_factor
        if not isinstance(self._param_groups[0], dict):
            params_grads = []
            lora_num = len(self._param_groups) // 2
            for i in range(lora_num):
                # 先转置
                A = self._param_groups[2 * i].detach().T
                B = self._param_groups[2 * i + 1].detach().T
                grad_A_orin = self._param_groups[2 * i]._grad_ivar().T
                grad_B_orin = self._param_groups[2 * i + 1]._grad_ivar().T

                # 中间与torch保持一致
                delta = 1e-8
                AA_T = A @ A.T
                B_TB = B.T @ B
                AA_T_inv = paddle.linalg.pinv(AA_T + delta * paddle.eye(A.shape[0]))
                B_TB_inv = paddle.linalg.pinv(B_TB + delta * paddle.eye(A.shape[0]))

                X = paddle.zeros((B_TB_inv.shape[0], B_TB_inv.shape[0])).cast(B.dtype)

                grad_A = (1 / scaling_factor**2) * B_TB_inv @ grad_A_orin + X @ A
                grad_B = (1 / scaling_factor**2) * (
                    (paddle.eye(B.shape[0]) - B @ B_TB_inv @ B.T) @ grad_B_orin @ AA_T_inv
                ) - B @ X

                # 最后转置回来
                self._param_groups[2 * i]._grad_ivar()[:] = grad_A.T
                self._param_groups[2 * i + 1]._grad_ivar()[:] = grad_B.T

            for param in self._param_groups:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))

            self._apply_optimize(
                loss=None,
                startup_program=None,
                params_grads=params_grads,
                param_group_idx=0,
            )

        else:
            # optimize parameters in groups
            for idx, param_group in enumerate(self._param_groups):
                params_grads = defaultdict(lambda: [])
                for param in param_group["params"]:
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        params_grads["params"].append((param, grad_var))
                params_grads.update({k: v for k, v in param_group.items() if k != "params"})
                self._apply_optimize(
                    loss=None,
                    startup_program=None,
                    params_grads=params_grads,
                    param_group_idx=idx,
                )
