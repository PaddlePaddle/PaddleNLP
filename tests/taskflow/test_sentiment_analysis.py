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

import os
import shutil
import unittest

from paddlenlp.taskflow import Taskflow


class TestSentimentAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # TODO rmtree will remove in the future
        if os.path.exists("__internal_testing__/tiny-random-skep"):
            shutil.rmtree("__internal_testing__/tiny-random-skep")

        cls.senta = Taskflow(
            task="sentiment_analysis",
            model="skep_ernie_1.0_large_ch",
            task_path="__internal_testing__/tiny-random-skep",
        )

    def test_single(self):
        input_text = ["蛋糕味道不错"]
        output_text = self.senta(input_text)
        self.assertTrue(len(output_text) == 1)
        self.assertIsInstance(output_text[0], dict)

    def test_batch(self):
        input_texts = ["蛋糕味道不错", "服务很好", "环境很差", "味道很香"]
        output_texts = self.senta(input_texts)
        self.assertTrue(len(output_texts) == len(input_texts))
        self.assertIsInstance(output_texts[0], dict)


if __name__ == "__main__":
    unittest.main()
