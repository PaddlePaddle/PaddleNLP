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

import json
import os
import random
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import paddle
from parameterized import parameterized

from paddlenlp.dataaug import WordDelete, WordInsert, WordSubstitute, WordSwap


class TestWordAug(unittest.TestCase):
    def setUp(self):
        self.sequences = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
        self.types = ["custom", "random", "mlm"]

        custom_dict = {"人类": ["人累", "扔雷"], "抽象": ["丑相"], "符号": ["富豪", "负号", "付豪"]}
        self.temp_dir = TemporaryDirectory()
        self.custom_file_path = os.path.join(self.temp_dir.name, "custom.json")
        with open(self.custom_file_path, "w", encoding="utf-8") as f:
            json.dump(custom_dict, open(self.custom_file_path, "w", encoding="utf-8"))
        f.close()

        self.seed = 42
        self.set_random_seed(self.seed)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    @parameterized.expand([(1,), (2,)])
    def test_word_substitute(self, create_n):
        for t in self.types:
            if t == "mlm":
                aug = WordSubstitute(
                    "mlm", create_n=create_n, model_name="__internal_testing__/tiny-random-ernie", vocab="test_vocab"
                )
                augmented = aug.augment(self.sequences)
                self.assertEqual(len(self.sequences), len(augmented))
                continue
            elif t == "custom":
                aug = WordSubstitute(
                    "custom", create_n=create_n, custom_file_path=self.custom_file_path, vocab="test_vocab"
                )
            else:
                aug = WordSubstitute(t, create_n=create_n, vocab="test_vocab")

            augmented = aug.augment(self.sequences)
            self.assertEqual(len(self.sequences), len(augmented))
            self.assertEqual(create_n, len(augmented[0]))
            self.assertEqual(create_n, len(augmented[1]))

    @parameterized.expand([(1,), (2,)])
    def test_word_insert(self, create_n):
        for t in self.types:
            if t == "mlm":
                aug = WordInsert(
                    "mlm", create_n=create_n, model_name="__internal_testing__/tiny-random-ernie", vocab="test_vocab"
                )
                augmented = aug.augment(self.sequences)
                self.assertEqual(len(self.sequences), len(augmented))
                continue
            elif t == "custom":
                aug = WordInsert(
                    "custom", create_n=create_n, custom_file_path=self.custom_file_path, vocab="test_vocab"
                )
            else:
                aug = WordInsert(t, create_n=create_n, vocab="test_vocab")

            augmented = aug.augment(self.sequences)
            self.assertEqual(len(self.sequences), len(augmented))
            self.assertEqual(create_n, len(augmented[0]))
            self.assertEqual(create_n, len(augmented[1]))

    @parameterized.expand([(1,)])
    def test_word_delete(self, create_n):
        aug = WordDelete(create_n=create_n, vocab="test_vocab")
        augmented = aug.augment(self.sequences)
        self.assertEqual(len(self.sequences), len(augmented))
        self.assertEqual(create_n, len(augmented[0]))
        self.assertEqual(create_n, len(augmented[1]))

    @parameterized.expand([(1,)])
    def test_word_swap(self, create_n):
        aug = WordSwap(create_n=create_n, vocab="test_vocab")
        augmented = aug.augment(self.sequences)
        self.assertEqual(len(self.sequences), len(augmented))
        self.assertEqual(create_n, len(augmented[0]))
        self.assertEqual(create_n, len(augmented[1]))


if __name__ == "__main__":
    unittest.main()
