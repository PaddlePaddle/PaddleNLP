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
from typing import List, Tuple

import paddle


class SequenceAccuracy(paddle.metric.Metric):
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


def wordseg_hard_acc(list_a: List[Tuple[str, str]],
                     list_b: List[Tuple[str, str]]) -> float:
    """
    Calculate extra metrics of word-seg

    Args:
        list_a: prediction list
        list_b: real list

    Returns:
        acc: the extra accuracy
    """
    p, q = 0, 0
    a_l, b_l = 0, 0
    acc = 0.0
    while q < len(list_b) and p < len(list_a):
        a_r = a_l + len(list_a[p][0]) - 1
        b_r = b_l + len(list_b[q][0]) - 1
        if a_r < b_l:
            p += 1
            a_l = a_r + 1
            continue
        if b_r < a_l:
            q += 1
            b_l = b_r + 1
            continue
        if a_l == b_l and a_r == b_r:
            acc += 1.0
            p += 1
            q += 1
            a_l = a_r + 1
            b_l = b_r + 1
            continue
        p += 1
    return acc


def wordtag_hard_acc(list_a: List[Tuple[str, str]],
                     list_b: List[Tuple[str, str]]) -> float:
    """
    Calculate extra metrics of word-tag

    Args:
        list_a: prediction list
        list_b: real list

    Returns:
        acc: the extra accuracy
    """
    p, q = 0, 0
    a_l, b_l = 0, 0
    acc = 0.0
    while q < len(list_b) and p < len(list_a):
        a_r = a_l + len(list_a[p][0]) - 1
        b_r = b_l + len(list_b[q][0]) - 1
        if a_r < b_l:
            p += 1
            a_l = a_r + 1
            continue
        if b_r < a_l:
            q += 1
            b_l = b_r + 1
            continue
        if a_l == b_l and a_r == b_r:
            if list_a[p][-1] == list_b[q][-1]:
                acc += 1.0
            p += 1
            q += 1
            a_l, b_l = a_r + 1, b_r + 1
            continue
        p += 1
    return acc


def wordtag_soft_acc(list_a: List[Tuple[str, str]],
                     list_b: List[Tuple[str, str]]) -> float:
    """
    Calculate extra metrics of word-tag

    Args:
        list_a: prediction list
        list_b: real list

    Returns:
        acc: the extra accuracy
    """
    p, q = 0, 0
    a_l, b_l = 0, 0
    acc = 0.0
    while q < len(list_b) and p < len(list_a):
        a_r = a_l + len(list_a[p][0]) - 1
        b_r = b_l + len(list_b[q][0]) - 1
        if a_r < b_l:
            p += 1
            a_l = a_r + 1
            continue
        if b_r < a_l:
            q += 1
            b_l = b_r + 1
            continue
        if a_l == b_l and a_r == b_r:
            if list_a[p][-1] == list_b[q][-1]:
                acc += 1.0
            elif list_b[q][-1].startswith(list_a[p][-1]):
                acc += 1.0
            elif list_b[q] == "词汇用语":
                acc += 1.0
            p += 1
            q += 1
            a_l, b_l = a_r + 1, b_r + 1
            continue
        p += 1
    return acc


def wordseg_soft_acc(list_a: List[Tuple[str, str]],
                     list_b: List[Tuple[str, str]]) -> float:
    """
    Calculate extra metrics of word-seg

    Args:
        list_a: prediction list
        list_b: real list

    Returns:
        acc: the extra accuracy
    """
    i, j = 0, 0
    acc = 0.0
    a_l, b_l = 0, 0
    while i < len(list_a) and j < len(list_b):
        a_r = a_l + len(list_a[i][0]) - 1
        b_r = b_l + len(list_b[j][0]) - 1
        if a_r < b_l:
            i += 1
            a_l = a_r + 1
            continue
        if b_r < a_l:
            j += 1
            b_l = b_r + 1
            continue
        if a_l == b_l and a_r == b_r:
            acc += 1.0
            a_l, b_l = a_r + 1, b_r + 1
            i, j = i + 1, j + 1
            continue
        if a_l == b_l and a_r < b_r:
            cnt = 0.0
            tmp_a_r = a_r
            for k in range(i + 1, len(list_a)):
                tmp_a_r += len(list_a[k])
                cnt += 1.0
                if tmp_a_r == b_r:
                    acc += cnt
                    i, j = k + 1, j + 1
                    a_l, b_l = tmp_a_r + 1, b_r + 1
                    break
            i += 1
            continue
        if a_l == b_l and a_r > b_r:
            tmp_b_r = b_r
            for k in range(j + 1, len(list_b)):
                tmp_b_r += len(list_b[k])
                if tmp_b_r == a_r:
                    acc += 1.0
                    i, j = i + 1, k + 1
                    a_l, b_l = a_r + 1, tmp_b_r + 1
                break
            j += 1
            continue
        i += 1
    return acc
