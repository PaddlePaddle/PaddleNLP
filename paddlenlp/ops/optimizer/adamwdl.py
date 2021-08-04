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
from functools import partial

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.optimizer import AdamW

__all__ = ['AdamWDL', ]


# Layerwise decay
def set_param_lr(decay_rate, name_dict, n_layers, param):
    """
        decay_rate: Layer-wise decay ratio.
        name_dict: static name of model, get from model.named_parameters().
        n_layers: total number of layers in the transformer encoder.
    """
    ratio = 1.0
    static_name = name_dict[param.name]
    if "encoder.layers" in static_name:
        idx = static_name.find("encoder.layers.")
        layer = int(static_name[idx:].split(".")[2])
        ratio = decay_rate**(n_layers - layer)
    elif "embedding" in static_name:
        ratio = decay_rate**(n_layers + 1)
    param.optimize_attr["learning_rate"] *= ratio


class AdamWDL(AdamW):
    """
    AdamW with dynamic lr setting
    “Layer-wise decay” means exponentially decaying the learning rates of individual 
    layers in a top-down manner. For example, suppose the 24-th layer uses a learning
    rate l, and the Layer-wise decay rate is α, then the learning rate of layer m 
    is lα^(24-m). See more details on: https://arxiv.org/abs/1906.08237
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
                 set_param_lr_fun=set_param_lr,
                 name_dict=None,
                 name=None):
        if not isinstance(layerwise_decay, float) and \
                not isinstance(layerwise_decay, fluid.framework.Variable):
            raise TypeError("coeff should be float or Tensor.")
        self.layerwise_decay = layerwise_decay
        self.n_layers = n_layers
        self.set_param_lr_fun = partial(set_param_lr_fun, layerwise_decay,
                                        name_dict, n_layers)
        super(AdamWDL, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            grad_clip=grad_clip,
            name=name,
            apply_decay_param_fun=apply_decay_param_fun,
            weight_decay=weight_decay,
            lazy_mode=lazy_mode,
            multi_precision=multi_precision)

    def _append_optimize_op(self, block, param_and_grad):
        if self.set_param_lr_fun is None:
            return super(AdamLW, self)._append_optimize_op(block,
                                                           param_and_grad)

        self._append_decoupled_weight_decay(block, param_and_grad)
        prev_lr = param_and_grad[0].optimize_attr["learning_rate"]
        self.set_param_lr_fun(param_and_grad[0])
        # excute Adam op
        res = super(AdamW, self)._append_optimize_op(block, param_and_grad)
        param_and_grad[0].optimize_attr["learning_rate"] = prev_lr
        return res
