# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# The file has been adapted from lightning file:
# https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/module.py
# Git commit hash: 2d9e00fab64c8b19a8646f755a95bcb092aa710f
# We retain the following license from the original files:

# Copyright 2018-2021 William Falcon. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import paddle.nn as nn


class BasicModule(nn.Layer):
    """ """

    def __init__(self, configs, *args, **kwargs):
        self.configs = self.process_configs(configs)
        super().__init__(*args, **kwargs)
        self.model = self.get_model()

    def process_configs(self, configs):
        return configs

    def get_model(self):
        raise NotImplementedError

    def get_loss_fn(self):
        pass

    def pretreating_batch(self, batch):
        return batch

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    def training_step_end(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def validation_step_end(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def test_step_end(self, *args, **kwargs):
        pass

    def backward(self, loss):
        loss.backward()

    def input_spec(self):
        raise NotImplementedError("Please redefine Module.input_spec for model export")

    def inference_end(self, outputs):
        pass

    def training_epoch_end(self, *args, **kwargs):
        pass

    def validation_epoch_end(self, *args, **kwargs):
        pass
