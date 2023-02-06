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

import os
import unittest
from tempfile import TemporaryDirectory

from parameterized import parameterized

from paddlenlp.taskflow import Taskflow
from paddlenlp.taskflow.fill_mask import FillMaskTask
from paddlenlp.transformers import AutoTokenizer, ErnieForMaskedLM


class TestFillMaskTask(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "model")
        model = ErnieForMaskedLM.from_pretrained("__internal_testing__/tiny-random-ernie")
        tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/tiny-random-ernie")
        model.save_pretrained(self.model_path)
        tokenizer.save_pretrained(self.model_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fill_mask_taskflow_invalid_inputs(self):
        taskflow = FillMaskTask(task="fill_mask", task_path=self.model_path)

        with self.assertRaises(ValueError):
            taskflow((["飞桨深度学习框"],))
            taskflow((["飞[MASK]深度学[MASK]"],))

    @parameterized.expand([(1, 1), (2, 3)])
    def test_fill_mask_taskflow(self, batch_size: int, top_k: int):
        # input_text is a tuple to simulate the args passed from Taskflow to TextClassificationTask
        input_text = (["飞桨深度学习框[MASK]", "生活的真谛是[MASK]"],)
        taskflow = FillMaskTask(task="fill_mask", task_path=self.model_path, batch_size=batch_size, top_k=top_k)

        results = taskflow(input_text)
        self.assertEqual(len(results), len(input_text[0]))
        for result in results:
            self.assertEqual(len(result), top_k)

    @parameterized.expand([(1, 1), (2, 3)])
    def test_taskflow(self, batch_size: int, top_k: int):
        input_text = ["飞桨深度学习框[MASK]", "生活的真谛是[MASK]"]
        taskflow = Taskflow(task="fill_mask", task_path=self.model_path, batch_size=batch_size, top_k=top_k)

        results = taskflow(input_text)
        self.assertEqual(len(results), len(input_text))
        for result in results:
            self.assertEqual(len(result), top_k)
