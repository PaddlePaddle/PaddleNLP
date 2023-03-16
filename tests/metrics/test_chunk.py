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

from paddlenlp.metrics import ChunkEvaluator


class TestChunk(unittest.TestCase):
    def test_metrics(self):
        label_list = ["O", "B-Person", "I-Person"]
        evaluator = ChunkEvaluator(label_list)
        evaluator.reset()
        lengths = paddle.to_tensor([5])
        predictions = paddle.to_tensor([[0, 1, 2, 1, 2]])
        labels = paddle.to_tensor([[0, 1, 2, 1, 1]])
        num_infer_chunks, num_label_chunks, num_correct_chunks = evaluator.compute(
            lengths=lengths, predictions=predictions, labels=labels
        )
        evaluator.update(num_infer_chunks.numpy(), num_label_chunks.numpy(), num_correct_chunks.numpy())
        precision, recall, f1 = evaluator.accumulate()
        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 0.3333333333333333)
        self.assertEqual(f1, 0.4)
