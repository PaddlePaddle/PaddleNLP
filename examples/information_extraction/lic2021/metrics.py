# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class Accuracy():
    """
    Calculate Accuracy
    Can calculate topK accuracy by specialist topK parameter
    Example:
        >>> metrics = Accuracy(**)
        >>> for epoch in range(epochs):
        >>>     metrics.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metrics(logits, target)
        >>>         print(metrics.name(), metrics.value())
    """

    def __init__(self, top_k):
        super(Accuracy, self).__init__()
        self.top_k = top_k
        self.correct_k = 0
        self.total = 0

    def __call__(self, logits, target):
        _, pred = logits.topk(self.top_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        self.correct_k = correct[:self.top_k].view(-1).float().sum(0)
        self.total = target.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k) / self.total

    def name(self):
        return "accuracy"


class SequenceAccuracy():
    """
    Masked language model pre-train task accuracy.
    """

    def __init__(self, top_k=1):
        super(SequenceAccuracy, self).__init__()
        self.top_k = top_k
        self.correct_k = 0
        self.total = 0

    def __call__(self, logits, target, ignore_index):
        pred = paddle.argmax(logits, 1)
        active_acc = target.view(-1) != ignore_index
        active_pred = pred[active_acc]
        active_labels = target[active_acc]

        correct = active_pred.eq(active_labels)
        self.correct_k = correct.float().sum(0)
        self.total = active_labels.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k) / self.total

    def name(self):
        return "Masked Language Model Accuracy"


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Example:
        >>> loss = AverageMeter()
        >>> for step, batch in enumerate(train_data):
        >>>     pred = self.model(batch)
        >>>     raw_loss = self.metrics(pred, target)
        >>>     loss.update(raw_loss.item(), n = 1)
        >>> cur_loss = loss.avg
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / float(self.count)
