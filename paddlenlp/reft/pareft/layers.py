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

import paddle
import paddle.nn as nn


class LowRankRotateLayer(nn.Layer):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m):
        super().__init__()
        # n > m
        print("n,m", n, m)

        # weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal())
        # linear = paddle.nn.Linear(10, 15, weight_attr=weight_attr)
        self.weight = self.create_parameter(
            shape=[n, m],
            attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal()),
            is_bias=False,
        )
        # print(self.weight.T @ self.weight )

    def forward(self, x):
        return paddle.matmul(x.astype(self.weight.dtype), self.weight)


# # 示例用法
# n, m = 4096, 4  # 示例维度
# layer = LowRankRotateLayer(n, m)
# layer.to("gpu")
# x = paddle.randn([n]).to("gpu")  # 示例输入
# print(x)
# output = layer(x)
# print(output)
