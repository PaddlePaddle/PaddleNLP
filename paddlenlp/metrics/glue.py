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

import os
import sys
import math
from functools import partial

import numpy as np
import paddle
from paddle.metric import Metric, Accuracy, Precision, Recall

__all__ = ['AccuracyAndF1', 'Mcc', 'PearsonAndSpearman']


class AccuracyAndF1(Metric):
    """
    Encapsulates Accuracy, Precision, Recall and F1 metric logic.
    """

    def __init__(self,
                 topk=(1, ),
                 pos_label=1,
                 name='acc_and_f1',
                 *args,
                 **kwargs):
        super(AccuracyAndF1, self).__init__(*args, **kwargs)
        self.topk = topk
        self.pos_label = pos_label
        self._name = name
        self.acc = Accuracy(self.topk, *args, **kwargs)
        self.precision = Precision(*args, **kwargs)
        self.recall = Recall(*args, **kwargs)
        self.reset()

    def compute(self, pred, label, *args):
        self.label = label
        self.preds_pos = paddle.nn.functional.softmax(pred)[:, self.pos_label]
        return self.acc.compute(pred, label)

    def update(self, correct, *args):
        self.acc.update(correct)
        self.precision.update(self.preds_pos, self.label)
        self.recall.update(self.preds_pos, self.label)

    def accumulate(self):
        acc = self.acc.accumulate()
        precision = self.precision.accumulate()
        recall = self.recall.accumulate()
        if precision == 0.0 or recall == 0.0:
            f1 = 0.0
        else:
            # 1/f1 = 1/2 * (1/precision + 1/recall)
            f1 = (2 * precision * recall) / (precision + recall)
        return (
            acc,
            precision,
            recall,
            f1,
            (acc + f1) / 2, )

    def reset(self):
        self.acc.reset()
        self.precision.reset()
        self.recall.reset()
        self.label = None
        self.preds_pos = None

    def name(self):
        """
        Return name of metric instance.
        """
        return self._name


class Mcc(Metric):
    """
    Matthews correlation coefficient
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient.
    """

    def __init__(self, name='mcc', *args, **kwargs):
        super(Mcc, self).__init__(*args, **kwargs)
        self._name = name
        self.tp = 0  # true positive
        self.fp = 0  # false positive
        self.tn = 0  # true negative
        self.fn = 0  # false negative

    def compute(self, pred, label, *args):
        preds = paddle.argsort(pred, descending=True)[:, :1]
        return (preds, label)

    def update(self, preds_and_labels):
        preds = preds_and_labels[0]
        labels = preds_and_labels[1]
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        if isinstance(labels, paddle.Tensor):
            labels = labels.numpy().reshape(-1, 1)
        sample_num = labels.shape[0]
        for i in range(sample_num):
            pred = preds[i]
            label = labels[i]
            if pred == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if pred == label:
                    self.tn += 1
                else:
                    self.fn += 1

    def accumulate(self):
        if self.tp == 0 or self.fp == 0 or self.tn == 0 or self.fn == 0:
            mcc = 0.0
        else:
            # mcc = (tp*tn-fp*fn)/ sqrt(tp+fp)(tp+fn)(tn+fp)(tn+fn))
            mcc = (self.tp * self.tn - self.fp * self.fn) / math.sqrt(
                (self.tp + self.fp) * (self.tp + self.fn) *
                (self.tn + self.fp) * (self.tn + self.fn))
        return (mcc, )

    def reset(self):
        self.tp = 0  # true positive
        self.fp = 0  # false positive
        self.tn = 0  # true negative
        self.fn = 0  # false negative

    def name(self):
        """
        Return name of metric instance.
        """
        return self._name


class PearsonAndSpearman(Metric):
    """
    Pearson correlation coefficient
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    Spearman's rank correlation coefficient
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient.
    """

    def __init__(self, name='mcc', *args, **kwargs):
        super(PearsonAndSpearman, self).__init__(*args, **kwargs)
        self._name = name
        self.preds = []
        self.labels = []

    def update(self, preds_and_labels):
        preds = preds_and_labels[0]
        labels = preds_and_labels[1]
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        if isinstance(labels, paddle.Tensor):
            labels = labels.numpy()
        preds = np.squeeze(preds.reshape(-1, 1)).tolist()
        labels = np.squeeze(labels.reshape(-1, 1)).tolist()
        self.preds.append(preds)
        self.labels.append(labels)

    def accumulate(self):
        preds = [item for sublist in self.preds for item in sublist]
        labels = [item for sublist in self.labels for item in sublist]
        #import pdb; pdb.set_trace()
        pearson = self.pearson(preds, labels)
        spearman = self.spearman(preds, labels)
        return (
            pearson,
            spearman,
            (pearson + spearman) / 2, )

    def pearson(self, preds, labels):
        n = len(preds)
        #simple sums
        sum1 = sum(float(preds[i]) for i in range(n))
        sum2 = sum(float(labels[i]) for i in range(n))
        #sum up the squares
        sum1_pow = sum([pow(v, 2.0) for v in preds])
        sum2_pow = sum([pow(v, 2.0) for v in labels])
        #sum up the products
        p_sum = sum([preds[i] * labels[i] for i in range(n)])

        numerator = p_sum - (sum1 * sum2 / n)
        denominator = math.sqrt(
            (sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def spearman(self, preds, labels):
        preds_rank = self.get_rank(preds)
        labels_rank = self.get_rank(labels)

        total = 0
        n = len(preds)
        for i in range(n):
            total += pow((preds_rank[i] - labels_rank[i]), 2)
        spearman = 1 - float(6 * total) / (n * (pow(n, 2) - 1))
        return spearman

    def get_rank(self, raw_list):
        x = np.array(raw_list)
        r_x = np.empty(x.shape, dtype=int)
        y = np.argsort(-x)
        for i, k in enumerate(y):
            r_x[k] = i + 1
        return r_x

    def reset(self):
        self.preds = []
        self.labels = []

    def name(self):
        """
        Return name of metric instance.
        """
        return self._name
