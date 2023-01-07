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

import numpy as np
import paddle
from paddle.metric import Metric


class Accuracy(Metric):
    def __init__(self, topk=(1,), name=None, *args, **kwargs):
        super(Accuracy, self).__init__(*args, **kwargs)
        self.topk = topk
        self.maxk = max(topk)
        self._init_name(name)
        self.reset()

    def compute(self, logits, target, *args):
        logits, target = (
            logits.detach(),
            target.detach(),
        )
        preds = logits.argmax(axis=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            correct = 1
        else:
            correct = preds == target
        return paddle.cast(correct, dtype="float32")

    def update(self, correct, *args):
        if isinstance(correct, (paddle.Tensor, paddle.fluid.core.eager.Tensor)):
            correct = correct.numpy()
        num_samples = np.prod(np.array(correct.shape[:-1]))
        accs = []
        for i, k in enumerate(self.topk):
            num_corrects = correct[..., :k].sum()
            accs.append(float(num_corrects) / num_samples)
            self.total[i] += num_corrects
            self.count[i] += num_samples
        accs = accs[0] if len(self.topk) == 1 else accs
        return accs

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.total = [0.0] * len(self.topk)
        self.count = [0] * len(self.topk)

    def accumulate(self):
        """
        Computes and returns the accumulated metric.
        """
        res = []
        for t, c in zip(self.total, self.count):
            r = float(t) / c if c > 0 else 0.0
            res.append(r)
        res = res[0] if len(self.topk) == 1 else res
        return res

    def _init_name(self, name):
        name = name or "acc"
        if self.maxk != 1:
            self._name = ["{}_top{}".format(name, k) for k in self.topk]
        else:
            self._name = [name]

    def name(self):
        """
        Return name of metric instance.
        """
        return self._name


class Scalar(Metric):
    def __init__(self, name="loss", *args, **kwargs):
        super(Scalar, self).__init__(*args, **kwargs)
        self._name = name
        self.reset()

    def update(self, scalar):
        self.scalar += scalar
        self.total += 1

    def accumulate(self):
        if self.total == 0:
            return paddle.zeros([1])
        return self.scalar / self.total

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.total = 0.0
        self.scalar = 0

    def name(self):
        """
        Returns metric name
        """
        return self._name


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
