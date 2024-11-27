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


class SimpleContrastiveLoss(paddle.nn.Layer):
    def __init__(self, embedding_temperature: float = 0.02):
        super().__init__()
        self.embedding_temperature = embedding_temperature
        self.cross_entropy = paddle.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, q_reps, p_reps):
        scores = paddle.matmul(q_reps, p_reps.transpose([1, 0]))
        scores = scores / self.embedding_temperature

        group_size = p_reps.shape[0] // q_reps.shape[0]
        batch_size = q_reps.shape[0]

        target = paddle.arange(batch_size, dtype="int64")
        target = target * group_size

        loss = self.cross_entropy(scores, target)
        return loss
