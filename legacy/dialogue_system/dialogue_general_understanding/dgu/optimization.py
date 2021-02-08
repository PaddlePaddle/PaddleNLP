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
"""Optimization and learning rate scheduling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
from dgu.utils.fp16 import create_master_params_grads, master_param_to_train_param


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
                 scheduler='linear_warmup_decay',
                 use_fp16=False,
                 loss_scaling=1.0,
                 clip_norm_thres=1.0):
    # When using mixed precision training, scale the gradient clip threshold
    # by loss_scaling
    if use_fp16 and loss_scaling > 1.0:
        clip_norm_thres *= loss_scaling

    if warmup_steps > 0:
        if scheduler == 'noam_decay':
            scheduled_lr = fluid.layers.learning_rate_scheduler\
             .noam_decay(1/(warmup_steps *(learning_rate ** 2)),
                         warmup_steps)
        elif scheduler == 'linear_warmup_decay':
            scheduled_lr = linear_warmup_decay(learning_rate, warmup_steps,
                                               num_train_steps)
        else:
            raise ValueError("Unkown learning rate scheduler, should be "
                             "'noam_decay' or 'linear_warmup_decay'")
        optimizer = fluid.optimizer.AdamOptimizer(
            learning_rate=scheduled_lr,
            grad_clip=fluid.clip.GradientClipByGlobalNorm(
                clip_norm=clip_norm_thres))
    else:
        optimizer = fluid.optimizer.AdamOptimizer(
            learning_rate=learning_rate,
            grad_clip=fluid.clip.GradientClipByGlobalNorm(
                clip_norm=clip_norm_thres))
        scheduled_lr = learning_rate

    def exclude_from_weight_decay(name):
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    param_list = dict()

    if use_fp16:
        param_grads = optimizer.backward(loss)
        master_param_grads = create_master_params_grads(
            param_grads, train_program, startup_prog, loss_scaling)

        for param, _ in master_param_grads:
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

        optimizer.apply_gradients(master_param_grads)

        if weight_decay > 0:
            for param, grad in master_param_grads:
                if exclude_from_weight_decay(param.name.rstrip(".master")):
                    continue
                with param.block.program._optimized_guard(
                    [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)

        master_param_to_train_param(master_param_grads, param_grads,
                                    train_program)

    else:
        for param in train_program.global_block().all_parameters():
            param_list[param.name] = param * 1.0
            param_list[param.name].stop_gradient = True

        _, param_grads = optimizer.minimize(loss)

        if weight_decay > 0:
            for param, grad in param_grads:
                if exclude_from_weight_decay(param.name):
                    continue
                with param.block.program._optimized_guard(
                    [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)

    return scheduled_lr
