# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.optimizer import AdamW


class AdamLW(AdamW):
    """
    Adam with layerwise decay and weight decay
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameters=None,
                 weight_decay=0.01,
                 apply_decay_param_fun=None,
                 grad_clip=None,
                 lazy_mode=False,
                 multi_precision=False,
                 layerwise_decay=0,
                 n_layers=12,
                 name=None):
        if not isinstance(layerwise_decay, float) and \
                not isinstance(layerwise_decay, fluid.framework.Variable):
            raise TypeError("coeff should be float or Tensor.")
        self.layerwise_decay = layerwise_decay
        self.n_layers = n_layers
        super(AdamLW, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            grad_clip=grad_clip,
            name=name,
            apply_decay_param_fun=apply_decay_param_fun,
            weight_decay=0.01,
            lazy_mode=lazy_mode,
            multi_precision=multi_precision)

    def _append_decoupled_layerwise_and_weight_decay(self, block,
                                                     param_and_grad):
        """
        Add decoupled weight decay op.
            parameter = parameter - parameter * coeff * lr

        Args:
            block: block in which variable is to be created
            param_and_grad: (parameters, gradients) pairs,
                the parameters need to decay.
        Raises:
            Exception: The type of coeff and parameter is not consistent.
        """
        param, grad = param_and_grad

        if isinstance(self._learning_rate, float):
            learning_rate = self._learning_rate
        else:
            # NOTE. We add this function to the _append_optimize_op(),
            # for we must make sure _create_param_lr() be called after
            # optimizer._create_global_learning_rate().
            learning_rate = self._create_param_lr(param_and_grad)

        with block.program._optimized_guard(
            [param, grad]), fluid.framework.name_scope('weight decay'):
            self._params_name.add(param.name)

            layer_decay_coeff = 1.0
            if self.layerwise_decay > 0:
                layer_decay_coeff = self._layer_decay_coeff(param)
            coeff = self._coeff
            if self._apply_decay_param_fun is not None \
                and not self._apply_decay_param_fun(param.name):
                coeff = 1.0

            decay_coeff = 1.0 - learning_rate * coeff * layer_decay_coeff

            find_master = (self._multi_precision and
                           param.dtype == core.VarDesc.VarType.FP16)
            if find_master:
                master_weight = self._master_weights[param.name]
                scaled_param = master_weight * decay_coeff
                paddle.fluid.layers.assign(
                    input=scaled_param, output=master_weight)
            else:
                scaled_param = param * decay_coeff
                paddle.fluid.layers.assign(input=scaled_param, output=param)

    def _append_optimize_op(self, block, param_and_grad):
        self._append_decoupled_layerwise_and_weight_decay(block, param_and_grad)
        # use Adam _append_optimize_op instead of AdamW
        return super(AdamW, self)._append_optimize_op(block, param_and_grad)

    def _layer_decay_coeff(self, param):
        """layerwise learning rate decay"""
        decay_rate = self.layerwise_decay
        n_layers = self.n_layers
        ratio = 1.0
        if "encoder_layer" in param.name:
            idx = param.name.find("encoder_layer")
            layer = int(param.name[idx:].split("_")[2].split(".")[0])
            ratio = decay_rate**(n_layers - layer)
        elif "embedding" in param.name:
            ratio = decay_rate**(n_layers + 1)
        return ratio
