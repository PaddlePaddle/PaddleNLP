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

from paddlenlp.metrics import ChunkEvaluator


class TestChunk(unittest.TestCase):
    def test_metrics(self):
        num_infer_chunks = 10
        num_label_chunks = 9
        num_correct_chunks = 8

        label_list = [1, 1, 0, 0, 1, 0, 1]
        evaluator = ChunkEvaluator(label_list)
        evaluator.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
        precision, recall, f1 = evaluator.accumulate()
        self.assertEqual(precision, 0.8)
        self.assertEqual(recall, 0.8888888888888888)
        self.assertEqual(f1, 0.8421052631578948)
