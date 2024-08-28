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

import math

import paddle
from paddle import nn


class dynamic(nn.Layer):
    def __init__(
        self,
        maximum_rank: int = 1,
    ):
        super(dynamic, self).__init__()
        self.maximum_rank = maximum_rank

        self.frozen = False
        self.current_rank = 0

    def get_dimension(self):
        return self.maximum_rank

    def get_rank(self):
        return self.current_rank

    def set_rank(self, rank, frozen=False):
        self.current_rank = max(0, min(rank, self.get_dimension()))
        self.frozen = frozen

    def forward(self, inputs, mode: bool = False):
        if self.training or mode:
            if self.frozen:
                pr = inputs[:, : self.get_rank()].detach()
                r = inputs[:, self.get_rank()]

                if len(r.shape) == 1:
                    r = r.unsqueeze(-1)
                result = paddle.concat([pr, r], axis=-1)

                return result * math.sqrt(self.get_dimension() / (self.get_rank() + 1))
            else:
                return inputs[:, : self.get_rank() + 1] * math.sqrt(self.get_dimension() / (self.get_rank() + 1))

        else:
            return inputs[:, : self.get_rank() + 1] * math.sqrt(self.get_dimension() / (self.get_rank() + 1))
