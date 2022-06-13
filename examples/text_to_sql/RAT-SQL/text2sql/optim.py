#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import traceback
import logging
import json
import re

import paddle

param_name_to_exclue_from_weight_decay = re.compile(
    r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')


def get_warmup_and_linear_decay(max_steps, warmup_steps):
    """ERNIE/demo/utils.py"""
    return lambda step: min(
        step / warmup_steps, 1. - (step - warmup_steps) /
        (max_steps - warmup_steps))


def init_optimizer(model, config, train_steps, scale_params_lr=None):
    if scale_params_lr is not None:
        for model, lr_scale in scale_params_lr:
            for param in model.parameters():
                param.optimize_attr['learning_rate'] *= lr_scale

    warmup_steps = int(config.warmup_proportion * train_steps)
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        config.learning_rate,
        get_warmup_and_linear_decay(train_steps, warmup_steps))
    optimizer = paddle.optimizer.AdamW(
        lr_scheduler,
        parameters=model.parameters(),
        weight_decay=config.weight_decay,
        apply_decay_param_fun=lambda n:
        not param_name_to_exclue_from_weight_decay.match(n),
        grad_clip=paddle.nn.ClipGradByGlobalNorm(config.grad_clip))
    return optimizer


if __name__ == "__main__":
    """run some simple test cases"""
    import types
    model = paddle.vision.models.LeNet()
    config = types.SimpleNamespace(learning_rate=1e-3,
                                   warmup_proportion=0.1,
                                   weight_decay=0.2,
                                   grad_clip=1.0)
    optim = init_optimizer(model, config, train_steps=10000)
    print(optim)
