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

import random
import unittest

import numpy as np

from paddlenlp.metrics import MRR
from tests.common_test import CommonTest


class TestMRR(CommonTest):
    def setUp(self):
        self.distance = "cosine"
        self.mrr = MRR(distance=self.distance)
        self.label_num = 10
        self.label_shape = (20,)
        self.embedding_shape = (20, 128)

    def get_random_case(self):
        labels = np.random.randint(0, self.label_num, size=self.label_shape).astype("int64")
        embeddings = np.random.uniform(0.1, 1.0, self.embedding_shape).astype("float64")
        all_distance = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]
        distance = random.choice(all_distance)
        return labels, embeddings, distance, all_distance

    def get_true_mrr_case(self):
        labels = np.array([1, 2, 1]).astype("int64")
        embeddings = np.array(
            [
                # cosine similarity: 1,2 => 0.991; 1,3=>0.851; 2,3=>0.912
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 4.0],
                [1.0, 100.0, 1000.0],
            ]
        )
        distance = "cosine"
        true_mrr = (1.0 / 2 + 0 + 1.0 / 2) / 3
        return labels, embeddings, distance, true_mrr

    def test_reset_distance(self):
        _, _, distance, _ = self.get_random_case()
        self.mrr.reset_distance(distance)
        self.check_output_equal(self.mrr.distance, distance)

    def test_compute_matrix_mrr(self):
        step = 100
        for i in range(step):
            labels, embeddings, distance, _ = self.get_random_case()
            self.mrr.reset_distance(distance)
            self.mrr.compute_matrix_mrr(labels, embeddings)

    def test_compute_true_mrr(self):
        labels, embeddings, distance, true_mrr = self.get_true_mrr_case()
        self.mrr.reset_distance(distance)
        mrr = self.mrr.compute_matrix_mrr(labels, embeddings)
        self.check_output_equal(mrr, true_mrr)


if __name__ == "__main__":
    unittest.main()
