# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle

from paddlenlp.metrics import SpanEvaluator


class TestSpanEvaluator(unittest.TestCase):
    def test_metrics(self):
        metric = SpanEvaluator()
        metric.reset()
        start_prob = paddle.to_tensor([[0.1, 0.1, 0.6, 0.2], [0.0, 0.9, 0.1, 0.0]])
        end_prob = paddle.to_tensor([[0.1, 0.1, 0.2, 0.6], [0.0, 0.9, 0.1, 0.0]])
        start_ids = paddle.to_tensor([[0, 0, 1, 0], [0, 0, 1, 0]])
        end_ids = paddle.to_tensor([[0, 0, 0, 1], [0, 0, 1, 0]])
        num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)
        precision, recall, f1 = metric.accumulate()
        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 0.5)
        self.assertEqual(f1, 0.5)
