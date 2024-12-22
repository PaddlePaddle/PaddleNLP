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

import paddle
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
