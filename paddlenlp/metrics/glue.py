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
    This class encapsulates Accuracy, Precision, Recall and F1 metric logic,
    and `accumulate` function returns accuracy, precision, recall and f1.
    The overview of all metrics could be seen at the document of `paddle.metric
    <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/metric/Overview_cn.html>`_ 
    for details.

    Args:
        topk (int or tuple(int), optional):
            Number of top elements to look at for computing accuracy.
            Defaults to (1,).
        pos_label (int, optional): The positive label for calculating precision
            and recall.
            Defaults to 1.
        name (str, optional):
            String name of the metric instance. Defaults to 'acc_and_f1'.

    Example:

        .. code-block::

            import paddle
            from paddlenlp.metrics import AccuracyAndF1

            x = paddle.to_tensor([[0.1, 0.9], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3]])
            y = paddle.to_tensor([[1], [0], [1], [1]])

            m = AccuracyAndF1()
            correct = m.compute(x, y)
            m.update(correct)
            res = m.accumulate()
            print(res) # (0.5, 0.5, 0.3333333333333333, 0.4, 0.45)

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
        """
        Accepts network's output and the labels, and calculates the top-k
        (maximum value in topk) indices for accuracy.

        Args:
            pred (Tensor): 
                Predicted tensor, and its dtype is float32 or float64, and
                has a shape of [batch_size, num_classes].
            label (Tensor):
                The ground truth tensor, and its dtype is is int64, and has a
                shape of [batch_size, 1] or [batch_size, num_classes] in one
                hot representation.

        Returns:
            Tensor: Correct mask, each element indicates whether the prediction
            equals to the label. Its' a tensor with a data type of float32 and
            has a shape of [batch_size, topk].

        """
        self.label = label
        self.preds_pos = paddle.nn.functional.softmax(pred)[:, self.pos_label]
        return self.acc.compute(pred, label)

    def update(self, correct, *args):
        """
        Updates the metrics states (accuracy, precision and recall), in order to
        calculate accumulated accuracy, precision and recall of all instances.

        Args:
            correct (Tensor):
                Correct mask for calculating accuracy, and it's a tensor with
                shape [batch_size, topk] and has a dtype of
                float32.

        """
        self.acc.update(correct)
        self.precision.update(self.preds_pos, self.label)
        self.recall.update(self.preds_pos, self.label)

    def accumulate(self):
        """
        Calculates and returns the accumulated metric.

        Returns:
            tuple: The accumulated metric. A tuple of shape (acc, precision,
            recall, f1, average_of_acc_and_f1)

            With the fields:

            - acc (numpy.float64):
                The accumulated accuracy.
            - precision (numpy.float64):
                The accumulated precision.
            - recall (numpy.float64):
                The accumulated recall.
            - f1 (numpy.float64):
                The accumulated f1.
            - average_of_acc_and_f1 (numpy.float64):
                The average of accumulated accuracy and f1.

        """
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
        """
        Resets all metric states.
        """
        self.acc.reset()
        self.precision.reset()
        self.recall.reset()
        self.label = None
        self.preds_pos = None

    def name(self):
        """
        Returns name of the metric instance.

        Returns:
           str: The name of the metric instance.

        """
        return self._name


class Mcc(Metric):
    """
    This class calculates `Matthews correlation coefficient <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_ .

    Args:
        name (str, optional):
            String name of the metric instance. Defaults to 'mcc'.

    Example:

        .. code-block::

            import paddle
            from paddlenlp.metrics import Mcc

            x = paddle.to_tensor([[-0.1, 0.12], [-0.23, 0.23], [-0.32, 0.21], [-0.13, 0.23]])
            y = paddle.to_tensor([[1], [0], [1], [1]])

            m = Mcc()
            (preds, label) = m.compute(x, y)
            m.update((preds, label))
            res = m.accumulate()
            print(res) # (0.0,)

    """

    def __init__(self, name='mcc', *args, **kwargs):
        super(Mcc, self).__init__(*args, **kwargs)
        self._name = name
        self.tp = 0  # true positive
        self.fp = 0  # false positive
        self.tn = 0  # true negative
        self.fn = 0  # false negative

    def compute(self, pred, label, *args):
        """
        Processes the pred tensor, and returns the indices of the maximum of each
        sample.

        Args:
            pred (Tensor):
                The predicted value is a Tensor with dtype float32 or float64.
                Shape is [batch_size, 1].
            label (Tensor):
                The ground truth value is Tensor with dtype int64, and its
                shape is [batch_size, 1].

        Returns:
            tuple: A tuple of preds and label. Each shape is
            [batch_size, 1], with dtype float32 or float64.

        """
        preds = paddle.argsort(pred, descending=True)[:, :1]
        return (preds, label)

    def update(self, preds_and_labels):
        """
        Calculates states, i.e. the number of true positive, false positive,
        true negative and false negative samples.

        Args:
            preds_and_labels (tuple[Tensor]):
                Tuple of predicted value and the ground truth label, with dtype
                float32 or float64. Each shape is [batch_size, 1].

        """
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
        """
        Calculates and returns the accumulated metric.

        Returns:
            tuple: The accumulated metric. A tuple of shape (mcc,)

            With the fields:

            - mcc (numpy.float64):
                The accumulated mcc.
                
        """
        if self.tp == 0 or self.fp == 0 or self.tn == 0 or self.fn == 0:
            mcc = 0.0
        else:
            # mcc = (tp*tn-fp*fn)/ sqrt(tp+fp)(tp+fn)(tn+fp)(tn+fn))
            mcc = (self.tp * self.tn - self.fp * self.fn) / math.sqrt(
                (self.tp + self.fp) * (self.tp + self.fn) *
                (self.tn + self.fp) * (self.tn + self.fn))
        return (mcc, )

    def reset(self):
        """
        Resets all metric states.
        """
        self.tp = 0  # true positive
        self.fp = 0  # false positive
        self.tn = 0  # true negative
        self.fn = 0  # false negative

    def name(self):
        """
        Returns name of the metric instance.

        Returns:
            str: The name of the metric instance.

        """
        return self._name


class PearsonAndSpearman(Metric):
    """
    The class calculates `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
    and `Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_ .


    Args:
        name (str, optional):
            String name of the metric instance. Defaults to 'pearson_and_spearman'.

    Example:

        .. code-block::

            import paddle
            from paddlenlp.metrics import PearsonAndSpearman

            x = paddle.to_tensor([[0.1], [1.0], [2.4], [0.9]])
            y = paddle.to_tensor([[0.0], [1.0], [2.9], [1.0]])

            m = PearsonAndSpearman()
            m.update((x, y))
            res = m.accumulate()
            print(res) # (0.9985229081857804, 1.0, 0.9992614540928901)

    """

    def __init__(self, name='pearson_and_spearman', *args, **kwargs):
        super(PearsonAndSpearman, self).__init__(*args, **kwargs)
        self._name = name
        self.preds = []
        self.labels = []

    def update(self, preds_and_labels):
        """
        Ensures the type of preds and labels is numpy.ndarray and reshapes them
        into [-1, 1].

        Args:
            preds_and_labels (tuple[Tensor] or list[Tensor]):
                Tuple of predicted value and the ground truth label, with dtype
                float32 or float64. Each shape is [batch_size, d0, ..., dN].

        """
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
        """
        Calculates and returns the accumulated metric.

        Returns:
            tuple: The accumulated metric. A tuple of (pearson, spearman,
            the_average_of_pearson_and_spearman)

            With the fields:

            - pearson (numpy.float64):
                The accumulated pearson.
            - spearman (numpy.float64):
                The accumulated spearman.
            - the_average_of_pearson_and_spearman (numpy.float64):
                The average of accumulated pearson and spearman correlation
                coefficient.

        """
        preds = [item for sublist in self.preds for item in sublist]
        labels = [item for sublist in self.labels for item in sublist]
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
        """
        Resets all metric states.
        """
        self.preds = []
        self.labels = []

    def name(self):
        """
        Returns name of the metric instance.

        Returns:
           str: The name of the metric instance.

        """
        return self._name
