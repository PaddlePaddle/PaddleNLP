# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

try:
    from collections.abc import Callable  # noqa
except ImportError:
    from collections import Callable  # noqa

import paddle
from paddle.fluid.framework import Variable
from paddle.fluid import framework
from paddle.fluid import layers
from paddle.fluid.layers import ops
from paddle.fluid import core
from paddle.fluid.optimizer import Optimizer

__all__ = ['AdamOptimizer', ]


class AdamOptimizer(Optimizer):
    r"""
    The Adam optimizer uses an optimization described at the end
    of section 2 of `Adam paper <https://arxiv.org/abs/1412.6980>`_ ,
    it can dynamically adjusts the learning rate of each parameter using
    the 1st moment estimates and the 2nd moment estimates of the gradient.

    The parameter ``param_out`` update rule with gradient ``grad``:
    .. math::
        t & = t + 1
        moment\_1\_out & = {\\beta}_1 * moment\_1 + (1 - {\\beta}_1) * grad
        moment\_2\_out & = {\\beta}_2 * moment\_2 + (1 - {\\beta}_2) * grad * grad
        learning\_rate & = learning\_rate * \\
                          \\frac{\sqrt{1 - {\\beta}_2^t}}{1 - {\\beta}_1^t}
        param\_out & = param - learning\_rate * \\frac{moment\_1}{\sqrt{moment\_2} + \epsilon}
    Related paper: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_
    Args:
        learning_rate (float|Variable, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a ``Variable`` with a float type. The default value is 0.001.
        beta1 (float|Variable, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a Variable with shape [1] and data type as float32.
            The default value is 0.9.
        beta2 (float|Variable, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a Variable with shape [1] and data type as float32.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-08.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.
        lazy_mode (bool, optional): The official Adam algorithm has two moving-average accumulators.
            The accumulators are updated at every step. Every element of the two moving-average
            is updated in both dense mode and sparse mode. If the size of parameter is very large,
            then the update may be very slow. The lazy mode only update the element that has
            gradient in current mini-batch, so it will be much more faster. But this mode has
            different semantics with the original Adam algorithm and may lead to different result.
            The default value is False.
    Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid
            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.data(name='x', shape=[None, 13], dtype='float32')
                y = fluid.data(name='y', shape=[None, 1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)
                adam_optimizer = fluid.optimizer.AdamOptimizer(0.01)
                adam_optimizer.minimize(avg_cost)
                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)
        .. code-block:: python
            # Adam with beta1/beta2 as Variable
            import paddle
            import paddle.fluid as fluid
            import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler
            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.data(name='x', shape=[None, 13], dtype='float32')
                y = fluid.data(name='y', shape=[None, 1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)
                # define beta decay variable
                def get_decayed_betas(beta1_init, beta2_init, decay_steps, decay_rate):
                    global_step = lr_scheduler._decay_step_counter()
                    beta1 = fluid.layers.create_global_var(
                        shape=[1],
                        value=float(beta1_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="beta1")
                    beta2 = fluid.layers.create_global_var(
                        shape=[1],
                        value=float(beta2_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="beta2")
                    div_res = global_step / decay_steps
                    decayed_beta1 = beta1_init * (decay_rate**div_res)
                    decayed_beta2 = beta2_init * (decay_rate**div_res)
                    fluid.layers.assign(decayed_beta1, beta1)
                    fluid.layers.assign(decayed_beta2, beta2)
                    return beta1, beta2
                beta1, beta2 = get_decayed_betas(0.9, 0.99, 1e5, 0.9)
                adam_optimizer = fluid.optimizer.AdamOptimizer(
                                                    learning_rate=0.01,
                                                    beta1=beta1,
                                                    beta2=beta2)
                adam_optimizer.minimize(avg_cost)
                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)
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
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 weight_decay=None,
                 lr_ratio=None,
                 apply_decay_param_fun=None,
                 name=None,
                 lazy_mode=False):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name)
        self.type = "custom_adam"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lazy_mode = lazy_mode
        self._weight_decay = weight_decay
        self._apply_decay_param_fun = apply_decay_param_fun
        if lr_ratio is not None:
            assert isinstance(lr_ratio, Callable)
        self._lr_ratio = lr_ratio

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            self._add_accumulator(self._moment1_acc_str, p)
            self._add_accumulator(self._moment2_acc_str, p)
            self._add_accumulator(
                name=self._beta1_pow_acc_str,
                param=p,
                fill_value=0.9 if isinstance(self._beta1, Variable) \
                        else self._beta1,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
            self._add_accumulator(
                name=self._beta2_pow_acc_str,
                param=p,
                fill_value=0.999 if isinstance(self._beta2, Variable) \
                        else self._beta2,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')

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
                type="custom_adam", inputs=ins, outputs=outs, attrs=attrs)

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
            "epsilon": self._epsilon,
            "lazy_mode": self._lazy_mode,
            "min_row_size_to_use_multithread": 1000,
            "weight_decay": weight_decay,
            "lr_ratio": 1.
            if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])
        }

        if isinstance(self._beta1, Variable):
            inputs['Beta1Tensor'] = self._beta1
        else:
            attrs['beta1'] = self._beta1
        if isinstance(self._beta2, Variable):
            inputs['Beta2Tensor'] = self._beta2
        else:
            attrs['beta2'] = self._beta2

        for name in ["Beta1Tensor", "Beta2Tensor", "MasterParam"]:
            if name in inputs:
                raise ValueError("Custom Adam should NOT have input: {}".format(
                    name))

        adam_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return adam_op
