# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class ExponentialMovingAverage(object):

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                self.shadow[name] = param.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay) * param + self.decay + self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                assert name in self.shadow
                self.backup[name] = param
                # TODO(huijuan): paddle中parameters赋值方式不是param.data，这样改不了模型参数
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                assert name in self.backup
                param = self.backup[name]
        self.backup = {}
