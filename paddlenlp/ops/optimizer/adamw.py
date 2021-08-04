# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
import os
import paddle
from paddle.optimizer.optimizer import Optimizer
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.framework import Variable
from paddle.fluid import layers
from paddle.fluid import unique_name
from paddle.fluid.framework import in_dygraph_mode, _dygraph_tracer
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import base as imperative_base

__all__ = ["AdamW", ]


class AdamW(Optimizer):
    r"""
    The AdamW optimizer is implemented based on the AdamW Optimization
    in paper `DECOUPLED WEIGHT DECAY REGULARIZATION <https://arxiv.org/pdf/1711.05101.pdf>`_.
    it can resolves the problem of L2 regularization failure in the Adam optimizer.
    .. math::
        t & = t + 1
        moment\_1\_out & = {\\beta}_1 * moment\_1 + (1 - {\\beta}_1) * grad
        moemnt\_2\_out & = {\\beta}_2 * moment\_2 + (1 - {\\beta}_2) * grad * grad
        learning\_rate & = learning\_rate * \\
            \\frac{\sqrt{1 - {\\beta}_2^t}}{1 - {beta}_1^t}
        param\_out & = param - learning\_rate * (\\frac{moment\_1}{\sqrt{moment\_2} + \epsilon} + \lambda * param)

    Args:
        learning_rate (float|LRScheduler, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a LRScheduler. The default value is 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            It should be a float number or a Tensor with shape [1] and data type as float32.
            The default value is 1e-08.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float, optional): The weight decay coefficient, it can be float or Tensor. The default value is 0.01.
        apply_decay_param_fun (function|None, optional): If it is not None,
            only tensors that makes apply_decay_param_fun(Tensor.name)==True
            will be updated. It only works when we want to specify tensors.
            Default: None.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        lazy_mode (bool, optional): The official Adam algorithm has two moving-average accumulators.
            The accumulators are updated at every step. Every element of the two moving-average
            is updated in both dense mode and sparse mode. If the size of parameter is very large,
            then the update may be very slow. The lazy mode only update the element that has
            gradient in current mini-batch, so it will be much more faster. But this mode has
            different semantics with the original Adam algorithm and may lead to different result.
            The default value is False.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating. Default is false.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Examples:
        .. code-block:: python

            import paddle
            import paddlenlp

            linear = paddle.nn.Linear(10, 10)
            inp = paddle.rand([10,10], dtype="float32")
            out = linear(inp)
            loss = paddle.mean(out)
            adamw = paddlenlp.ops.optimizer.Adam(learning_rate=0.1,
                    parameters=linear.parameters())
            out.backward()
            adamw.step()
            adamw.clear_grad()

    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameters=None,
                 weight_decay=0.0,
                 grad_clip=None,
                 lazy_mode=False,
                 multi_precision=False,
                 apply_decay_param_fun=None,
                 name=None):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        if not isinstance(beta1, Variable):
            if not 0 <= beta1 < 1:
                raise ValueError(
                    "Invaild value of beta1, expect beta1 in [0,1).")
        if not isinstance(beta2, Variable):
            if not 0 <= beta2 < 1:
                raise ValueError(
                    "Invaild value of beta2, expect beta2 in [0,1).")
        if not isinstance(epsilon, Variable):
            if not 0 <= epsilon:
                raise ValueError(
                    "Invaild value of epsilon, expect epsilon >= 0.")
        super(AdamW, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=None,
            grad_clip=grad_clip,
            name=name)
        self.type = "adamw"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lazy_mode = lazy_mode
        self._multi_precision = multi_precision
        self._weight_decay = weight_decay
        self._apply_decay_param_fun = apply_decay_param_fun
        self._master_weights = {}

    def _create_master_weight(self, param):
        assert isinstance(self.helper, LayerHelper)

        var_name = param.name + "_fp32_master"
        var_name = unique_name.generate(var_name)
        var = layers.create_global_var(
            name=var_name,
            shape=param.shape,
            value=0,
            dtype='float32',
            persistable=True)
        block = self.helper.startup_program.global_block()
        block.append_op(
            type="cast",
            inputs={"X": [param]},
            outputs={"Out": [var]},
            attrs={
                "in_dtype": param.dtype,
                "out_dtype": core.VarDesc.VarType.FP32
            })
        self._master_weights[param.name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter
        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched
        Returns:
            accumulator variable for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        find_master = self._multi_precision and param.dtype == core.VarDesc.VarType.FP16
        target_param = self._master_weights[
            param.name] if find_master else param
        target_name = target_param.name
        if (name not in self._accumulators or
                target_name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, target_name))
        return self._accumulators[name][target_name]

    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if acc_dtype == core.VarDesc.VarType.FP16:
            acc_dtype = core.VarDesc.VarType.FP32
        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype)
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.9 if isinstance(self._beta1, Variable) \
                    else self._beta1,
            shape=[1],
            type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.999 if isinstance(self._beta2, Variable) \
                    else self._beta2,
            shape=[1],
            type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if self._multi_precision and p.dtype == core.VarDesc.VarType.FP16:
                master_p = self._create_master_weight(p)
                self._add_moments_pows(master_p)
                continue
            if p.dtype == core.VarDesc.VarType.FP16 and not self._multi_precision:
                warnings.warn(
                    "Accumulating with FP16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Adam optimizer."
                )
            self._add_moments_pows(p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        beta2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                              param_and_grad[0])
        find_master = self._multi_precision and param_and_grad[
            0].dtype == core.VarDesc.VarType.FP16
        master_weight = (self._master_weights[param_and_grad[0].name]
                         if find_master else None)
        lr = self._create_param_lr(param_and_grad)

        # create the adam optimize op
        if self._apply_decay_param_fun is not None \
            and not self._apply_decay_param_fun(param_and_grad[0].name):
            weight_decay = 0.0
        else:
            weight_decay = self._weight_decay

        if framework.in_dygraph_mode():
            _beta1 = self._beta1 if not isinstance(
                self._beta1, Variable) else self._beta1.numpy().item(0)
            _beta2 = self._beta2 if not isinstance(
                self._beta2, Variable) else self._beta2.numpy().item(0)

            ins = {
                'Param': param_and_grad[0],
                'Grad': param_and_grad[1],
                'LearningRate': lr,
                'Moment1': moment1,
                'Moment2': moment2,
                'Beta1Pow': beta1_pow_acc,
                'Beta2Pow': beta2_pow_acc,
            }
            attrs = {
                'beta1': _beta1,
                'beta2': _beta2,
                'epsilon': self._epsilon,
                'lazy_mode': self._lazy_mode,
                'min_row_size_to_use_multithread': 1000,
                'multi_precision': False,
                'weight_decay': weight_decay,
                'lr_ratio': 1.0
            }
            outs = {
                'ParamOut': param_and_grad[0],
                'Moment1Out': moment1,
                'Moment2Out': moment2,
                'Beta1PowOut': beta1_pow_acc,
                'Beta2PowOut': beta2_pow_acc,
            }

            framework._dygraph_tracer().trace_op(
                type="adamw", inputs=ins, outputs=outs, attrs=attrs)

            return None

        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "LearningRate": [lr],
            "Moment1": [moment1],
            "Moment2": [moment2],
            "Beta1Pow": [beta1_pow_acc],
            "Beta2Pow": [beta2_pow_acc]
        }
        outputs = {
            "ParamOut": [param_and_grad[0]],
            "Moment1Out": [moment1],
            "Moment2Out": [moment2],
            "Beta1PowOut": [beta1_pow_acc],
            "Beta2PowOut": [beta2_pow_acc],
        }
        attrs = {
            "lazy_mode": self._lazy_mode,
            "min_row_size_to_use_multithread": 1000,
            "multi_precision": find_master,
            'weight_decay': weight_decay,
            'lr_ratio': 1.0
        }

        if isinstance(self._beta1, Variable):
            inputs['Beta1Tensor'] = self._beta1
        else:
            attrs['beta1'] = self._beta1
        if isinstance(self._beta2, Variable):
            inputs['Beta2Tensor'] = self._beta2
        else:
            attrs['beta2'] = self._beta2
        if isinstance(self._epsilon, Variable):
            inputs['EpsilonTensor'] = self._epsilon
        else:
            attrs['epsilon'] = self._epsilon

        if find_master:
            inputs["MasterParam"] = master_weight
            outputs["MasterParamOut"] = master_weight

        for name in ["Beta1Tensor", "Beta2Tensor", "MasterParam"]:
            if name in inputs:
                raise ValueError(
                    "Custom AdamW should NOT have input: {}".format(name))

        adam_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return adam_op

    @imperative_base.no_grad
    @framework.dygraph_only
    def step(self):
        params_grads = []
        for param in self._parameter_list:
            if param.stop_gradient:
                continue
            if param._grad_ivar() is not None:
                grad_var = param._grad_ivar()
                if hasattr(grad_var, "_is_sparse") and grad_var._is_sparse(
                ) and self.regularization is not None:
                    raise RuntimeError(
                        "AdamW don't support weight_decay with sparse parameters, please set it to None."
                    )
                params_grads.append((param, grad_var))

        optimize_ops = self._apply_optimize(
            loss=None, startup_program=None, params_grads=params_grads)
