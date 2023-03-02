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


class TestZeroShotTextClassificationTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = ["这是一条差评", "这是一条好评"]
        cls.taskflow = Taskflow(
            task="zero_shot_text_classification",
            model="__internal_testing__/tiny-random-utc",
            schema=cls.schema,
        )

    def test_single(self):
        output = self.taskflow("房间干净明亮，非常不错")
        self.assertEqual(len(output), 1)
        self.assertIn("text_a", output[0])
        self.assertIn("predictions", output[0])
        for pred in output[0]["predictions"]:
            self.assertIn(pred["label"], self.schema)

    def test_batch(self):
        outputs = self.taskflow(["房间干净明亮，非常不错", "这馆子不咋地"])
        self.assertEqual(len(outputs), 2)
        for output in outputs:
            self.assertIn("text_a", output)
            self.assertIn("predictions", output)
            for pred in output["predictions"]:
                self.assertIn(pred["label"], self.schema)

    def test_pair(self):
        output = self.taskflow([["测试句子1", "句子2"]])
        self.assertEqual(len(output), 1)
        self.assertIn("text_a", output[0])
        self.assertIn("predictions", output[0])
        for pred in output[0]["predictions"]:
            self.assertIn(pred["label"], self.schema)
