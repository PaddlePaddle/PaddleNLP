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
from paddlenlp.metrics import Perplexity

from util import stable_softmax, cross_entropy
from common_test import CommonTest


class NpPerplexity(object):
    def __init__(self):
        self.total_ce = 0
        self.total_word_num = 0

    def compute(self, pred, label, seq_mask=None):
        label = np.expand_dims(label, axis=2)
        ce = cross_entropy(
            softmax=pred,
            label=label,
            soft_label=False,
            axis=-1,
            ignore_index=-100)
        ce = np.squeeze(ce, axis=2)
        if seq_mask is not None:
            ce = ce * seq_mask
            word_num = np.sum(seq_mask)
            return ce, word_num
        return ce

    def update(self, ce):
        self.total_ce += np.sum(ce)
        self.total_word_num += ce.size

    def accumulate(self):
        return np.exp(self.total_ce / self.total_word_num)


class TestPerplexity(CommonTest):
    def setUp(self):
        self.config['name'] = 'test_perplexity'
        self.cls_num = 10
        self.shape = (5, 20, self.cls_num)
        self.label_shape = (5, 20)
        self.metrics = Perplexity(**self.config)
        self.np_metrics = NpPerplexity()

    def get_random_case(self):
        label = np.random.randint(
            self.cls_num, size=self.label_shape).astype("int64")
        logits = np.random.uniform(
            0.1, 1.0, self.shape).astype(paddle.get_default_dtype())
        pred = np.apply_along_axis(stable_softmax, -1, logits)
        seq_mask = np.random.randint(2, size=self.label_shape).astype("int64")
        return label, logits, pred, seq_mask

    def test_name(self):
        self.check_output_equal(self.metrics.name(), self.config['name'])

    def test_compute(self):
        label, logits, pred, _ = self.get_random_case()
        expected_result = self.np_metrics.compute(pred, label)
        result = self.metrics.compute(
            paddle.to_tensor(logits), paddle.to_tensor(label))
        self.check_output_equal(expected_result, result.numpy())

    def test_compute_with_mask(self):
        label, logits, pred, seq_mask = self.get_random_case()
        expected_result = self.np_metrics.compute(pred, label, seq_mask)
        result = self.metrics.compute(
            paddle.to_tensor(logits),
            paddle.to_tensor(label), paddle.to_tensor(seq_mask))
        self.check_output_equal(expected_result[0], result[0].numpy())
        self.check_output_equal(expected_result[1], result[1])

    def test_reset(self):
        label, logits, pred, _ = self.get_random_case()
        result = self.metrics.compute(
            paddle.to_tensor(logits), paddle.to_tensor(label))
        self.metrics.update(result.numpy())
        self.check_output_not_equal(self.metrics.total_ce, 0)
        self.check_output_not_equal(self.metrics.total_word_num, 0)

        self.metrics.reset()
        self.check_output_equal(self.metrics.total_ce, 0)
        self.check_output_equal(self.metrics.total_word_num, 0)

    def test_update_accumulate(self):
        steps = 10
        for i in range(steps):
            label, logits, pred, _ = self.get_random_case()
            expected_result = self.np_metrics.compute(pred, label)
            result = self.metrics.compute(
                paddle.to_tensor(logits), paddle.to_tensor(label))
            self.metrics.update(result.numpy())
            self.np_metrics.update(expected_result)
        self.check_output_equal(self.metrics.accumulate(),
                                self.np_metrics.accumulate())


if __name__ == "__main__":
    unittest.main()
