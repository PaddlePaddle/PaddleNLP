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
import unittest
import paddle
import os

__all__ = ['get_vocab_list', 'stable_softmax', 'cross_entropy']


def get_vocab_list(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_list = [
            vocab.rstrip("\n").split("\t")[0] for vocab in f.readlines()
        ]
        return vocab_list


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)

    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1:]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


def softmax_with_cross_entropy(logits,
                               label,
                               soft_label=False,
                               axis=-1,
                               ignore_index=-1):
    softmax = np.apply_along_axis(stable_softmax, -1, logits)
    return cross_entropy(softmax, label, soft_label, axis, ignore_index)


def assert_raises(Error=AssertionError):
    def assert_raises_error(func):
        def wrapper(self, *args, **kwargs):
            with self.assertRaises(Error):
                func(self, *args, **kwargs)

        return wrapper

    return assert_raises_error


def create_test_data(file=__file__):
    dir_path = os.path.dirname(os.path.realpath(file))
    test_data_file = os.path.join(dir_path, 'dict.txt')
    with open(test_data_file, "w") as f:
        vocab_list = [
            '[UNK]', 'AT&T', 'B超', 'c#', 'C#', 'c++', 'C++', 'T恤', 'A座', 'A股',
            'A型', 'A轮', 'AA制', 'AB型', 'B座', 'B股', 'B型', 'B轮', 'BB机', 'BP机',
            'C盘', 'C座', 'C语言', 'CD盒', 'CD机', 'CALL机', 'D盘', 'D座', 'D版', 'E盘',
            'E座', 'E化', 'E通', 'F盘', 'F座', 'G盘', 'H盘', 'H股', 'I盘', 'IC卡', 'IP卡',
            'IP电话', 'IP地址', 'K党', 'K歌之王', 'N年', 'O型', 'PC机', 'PH值', 'SIM卡',
            'U盘', 'VISA卡', 'Z盘', 'Q版', 'QQ号', 'RSS订阅', 'T盘', 'X光', 'X光线', 'X射线',
            'γ射线', 'T恤衫', 'T型台', 'T台', '4S店', '4s店', '江南style', '江南Style',
            '1号店', '小S', '大S', '阿Q', '一', '一一', '一一二', '一一例', '一一分', '一一列举',
            '一一对', '一一对应', '一一记', '一一道来', '一丁', '一丁不识', '一丁点', '一丁点儿', '一七',
            '一七八不', '一万', '一万一千', '一万一千五百二十颗', '一万一千八百八十斤', '一万一千多间',
            '一万一千零九十五册', '一万七千', '一万七千余', '一万七千多', '一万七千多户', '一万万'
        ]
        for vocab in vocab_list:
            f.write("{}\n".format(vocab))
    return test_data_file
