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


class TestDialogueTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogue = Taskflow(
            task="dialogue",
            model="__internal_testing__/tiny-random-plato",
        )
        cls.max_turn = 3

    def test_single(self):
        input_text = ["吃饭了吗"]
        for turn in range(self.max_turn):
            output_text = self.dialogue(input_text)
            self.assertTrue(len(output_text) == 1)
            self.assertIsInstance(output_text[0], str)
            input_text.append(output_text[0])

    def test_batch(self):
        input_text_1 = ["你好"]
        input_text_2 = ["你是谁？"]
        for turn in range(self.max_turn):
            output_text = self.dialogue(input_text_1, input_text_2)
            self.assertTrue(len(output_text) == 2)
            self.assertIsInstance(output_text[0], str)
            self.assertIsInstance(output_text[1], str)
            input_text_1.append(output_text[0])
            input_text_2.append(output_text[1])
