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

from paddlenlp.dataaug import CharDelete, CharInsert, CharSubstitute, CharSwap


class TestCharAug(unittest.TestCase):
    def setUp(self):
        self.sequences = ["人类语言是抽象的信息符号，其中蕴含着丰富的语义信息，人类可以很轻松地理解其中的含义。", "而计算机只能处理数值化的信息，无法直接理解人类语言，所以需要将人类语言进行数值化转换。"]
        self.types = ["custom", "random", "mlm"]

        custom_dict = {
            "人": ["任", "认", "忍"],
            "抽": ["丑", "臭"],
            "轻": ["亲", "秦"],
            "数": ["书", "树"],
            "转": ["赚", "专"],
            "理": ["里", "例"],
        }
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
    def test_char_substitute(self, create_n):
        for t in self.types:
            if t == "mlm":
                aug = CharSubstitute(
                    "mlm", create_n=create_n, model_name="__internal_testing__/tiny-random-ernie", vocab="test_vocab"
                )
                augmented = aug.augment(self.sequences)
                self.assertEqual(len(self.sequences), len(augmented))
                continue
            elif t == "custom":
                aug = CharSubstitute(
                    "custom", create_n=create_n, custom_file_path=self.custom_file_path, vocab="test_vocab"
                )
            else:
                aug = CharSubstitute(t, create_n=create_n, vocab="test_vocab")

            augmented = aug.augment(self.sequences)
            self.assertEqual(len(self.sequences), len(augmented))
            self.assertEqual(create_n, len(augmented[0]))
            self.assertEqual(create_n, len(augmented[1]))

    @parameterized.expand([(1,), (2,)])
    def test_char_insert(self, create_n):
        for t in self.types:
            if t == "mlm":
                aug = CharInsert(
                    "mlm", create_n=create_n, model_name="__internal_testing__/tiny-random-ernie", vocab="test_vocab"
                )
                augmented = aug.augment(self.sequences)
                self.assertEqual(len(self.sequences), len(augmented))
                continue
            elif t == "custom":
                aug = CharInsert(
                    "custom", create_n=create_n, custom_file_path=self.custom_file_path, vocab="test_vocab"
                )
            else:
                aug = CharInsert(t, create_n=create_n, vocab="test_vocab")

            augmented = aug.augment(self.sequences)
            self.assertEqual(len(self.sequences), len(augmented))
            self.assertEqual(create_n, len(augmented[0]))
            self.assertEqual(create_n, len(augmented[1]))

    @parameterized.expand([(1,)])
    def test_char_delete(self, create_n):
        aug = CharDelete(create_n=create_n, vocab="test_vocab")
        augmented = aug.augment(self.sequences)
        self.assertEqual(len(self.sequences), len(augmented))
        self.assertEqual(create_n, len(augmented[0]))
        self.assertEqual(create_n, len(augmented[1]))

    @parameterized.expand([(1,)])
    def test_char_swap(self, create_n):
        aug = CharSwap(create_n=create_n, vocab="test_vocab")
        augmented = aug.augment(self.sequences)
        self.assertEqual(len(self.sequences), len(augmented))
        self.assertEqual(create_n, len(augmented[0]))
        self.assertEqual(create_n, len(augmented[1]))


if __name__ == "__main__":
    unittest.main()
