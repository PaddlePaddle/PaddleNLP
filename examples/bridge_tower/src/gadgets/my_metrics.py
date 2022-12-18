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

import paddle
from paddle.metric import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=paddle.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=paddle.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += paddle.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=paddle.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=paddle.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, paddle.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = paddle.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self):
        super().__init__()
        self.score = 0.0
        self.total = 0.0

    def update(self, logits, target):
        logits, target = (
            logits,
            target,
        )
        logits = paddle.max(logits, 1)[1]
        # .to(target)
        one_hots = paddle.zeros(*target.shape)
        one_hots.scatter_(1, logits.reshape([-1, 1]), 1)
        scores = one_hots * target

        self.score += scores.sum().numpy()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total
