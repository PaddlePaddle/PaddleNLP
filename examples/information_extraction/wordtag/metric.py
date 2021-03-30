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

import paddle


class SequenceAccuracy():
    """
    Masked language model pre-train task accuracy.
    """

    def __init__(self):
        super(SequenceAccuracy, self).__init__()
        self.correct_k = 0
        self.total = 0

    def compute(self, pred, label, ignore_index):
        pred = paddle.argmax(pred, 1)
        active_acc = label.reshape([-1]) != ignore_index
        active_pred = pred.masked_select(active_acc)
        active_labels = label.masked_select(active_acc)

        correct = active_pred.equal(active_labels)
        return correct

    def update(self, correct):
        self.correct_k += correct.cast('float32').sum(0)
        self.total += correct.shape[0]

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def accumulate(self):
        return float(self.correct_k) / self.total

    def name(self):
        return "Masked Language Model Accuracy"
