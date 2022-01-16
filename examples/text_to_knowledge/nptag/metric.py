# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class NPTagAccuracy(paddle.metric.Metric):
    """
    Accuracy for NPTag Prompt Model.
    """

    def __init__(self):
        super(NPTagAccuracy, self).__init__()
        self.reset()

    def reset(self):
        self.corrects = 0
        self.total = 0

    def compute(self, preds, labels):
        correct = []
        for pred, label in zip(preds, labels):
            real_pred, real_label = ([] for _ in range(2))
            for i in range(len(label)):
                if label[i] == -100 or label[i] == 0:
                    continue
                real_pred.append(pred[i])
                real_label.append(label[i])

            if all(real_pred[i] == real_label[i]
                   for i in range(len(real_label))):
                correct.append(1)
            else:
                correct.append(0)
        return correct

    def update(self, correct):
        self.corrects += sum(correct)
        self.total += len(correct)

    def accumulate(self):
        return float(self.corrects) / self.total

    def name(self):
        return "NPTag Prompt Model Accuracy"
