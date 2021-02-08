#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Optimization and learning rate scheduling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import paddle.fluid as fluid


def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ Applies linear warmup of learning rate from 0 and decay to 0."""
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=learning_rate,
                    decay_steps=num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)

        return lr


def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 startup_prog,
                 weight_decay,
                 lr_layer_decay_rate=1.0,
                 scheduler='linear_warmup_decay'):

    scheduled_lr = None
    if scheduler == 'noam_decay':
        if warmup_steps > 0:
            scheduled_lr = fluid.layers.learning_rate_scheduler\
             .noam_decay(1/(warmup_steps *(learning_rate ** 2)),
           warmup_steps)
        else:
            printf(
                "WARNING: noam decay should have postive warmup steps, using "
                "constant learning rate instead!")
            scheduled_lr = fluid.layers.create_global_var(
                name=fluid.unique_name.generate("learning_rate"),
                shape=[1],
                value=learning_rate,
                dtype='float32',
                persistable=True)
    elif scheduler == 'linear_warmup_decay':
        scheduled_lr = linear_warmup_decay(learning_rate, warmup_steps,
                                           num_train_steps)
    else:
        raise ValueError("Unkown learning rate scheduler, should be "
                         "'noam_decay' or 'linear_warmup_decay'")

    if lr_layer_decay_rate != 1.0:
        n_layer = 0
        for param in fluid.default_main_program().block(0).all_parameters():
            m = re.search(r"model_transformer_layer_(\d+?)_", param.name)
            if not m: continue
            n_layer = max(n_layer, int(m.group(1)) + 1)

        for param in fluid.default_main_program().block(0).all_parameters():
            for l in range(n_layer):
                if "model_transformer_layer_{}_".format(l) in param.name:
                    param.optimize_attr[
                        'learning_rate'] = lr_layer_decay_rate**(
                            n_layer - 1 - l)
                    print("Apply lr decay {:.4f} to layer-{} grad of {}".format(
                        param.optimize_attr['learning_rate'], l, param.name))
                    break

    def exclude_from_weight_decay(param):
        name = param.name.rstrip(".master")
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)

    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))

    param_list = dict()

    if weight_decay > 0:
        for param in train_program.global_block().all_parameters():
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

    _, param_grads = optimizer.minimize(loss)

    if weight_decay > 0:
        for param, grad in param_grads:
            if exclude_from_weight_decay(param):
                continue
            with param.block.program._optimized_guard(
                [param, grad]), fluid.framework.name_scope("weight_decay"):
                updated_param = param - param_list[
                    param.name] * weight_decay * scheduled_lr
                fluid.layers.assign(output=param, input=updated_param)

    return scheduled_lr
