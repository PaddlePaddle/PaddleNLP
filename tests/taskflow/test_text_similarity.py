# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.taskflow import Taskflow


class TestTextSimilarityTask(unittest.TestCase):
    def test_bert_model(self):
        similarity = Taskflow(
            task="text_similarity",
            model="__internal_testing__/tiny-random-bert",
        )
        results = similarity([["世界上什么东西最小", "世界上什么东西最小？"]])
        self.assertTrue(len(results) == 1)
        self.assertTrue("text1" in results[0])
        self.assertTrue("text2" in results[0])
        self.assertIsInstance(results[0]["similarity"], float)

        results = similarity([["光眼睛大就好看吗", "眼睛好看吗？"], ["小蝌蚪找妈妈怎么样", "小蝌蚪找妈妈是谁画的"]])
        self.assertTrue(len(results) == 2)
        for result in results:
            self.assertTrue("text1" in result)
            self.assertTrue("text2" in result)
            self.assertIsInstance(result["similarity"], float)
