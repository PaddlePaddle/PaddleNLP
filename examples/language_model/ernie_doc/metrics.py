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


# Metric for ERNIE-DOC
class Acc(object):
    def __call__(self, preds, labels):
        if isinstance(preds, list):
            preds = np.array(preds, dtype='int64')
        if isinstance(labels, list):
            labels = np.array(labels, dtype='int64')
        acc = (preds == labels).sum() * 1.0 / labels.size
        return acc


class F1(object):
    def __init__(self, positive_label=1):
        self.positive_label = positive_label

    def __call__(self, preds, labels):
        if isinstance(preds, list):
            preds = np.array(preds, dtype='int64')
        if isinstance(labels, list):
            labels = np.array(labels, dtype='int64')
        tp = ((preds == labels) & (labels == self.positive_label)).sum()
        fn = ((preds != labels) & (labels == self.positive_label)).sum()
        fp = ((preds != labels) & (preds == self.positive_label)).sum()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * recall * precision / (recall + precision)
        return f1
