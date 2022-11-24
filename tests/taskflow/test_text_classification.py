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

import paddle
from parameterized import parameterized
from tempfile import TemporaryDirectory
from paddlenlp.taskflow import Taskflow
from paddlenlp.taskflow.text_classification import TextClassificationTask
from paddlenlp.transformers import AutoTokenizer, ErnieForSequenceClassification


class TestTextClassificationTask(unittest.TestCase):

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.dygraph_model_path = os.path.join(self.temp_dir.name, "dygraph")
        model = ErnieForSequenceClassification.from_pretrained(
            "__internal_testing__/ernie", num_classes=2)
        tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/ernie")
        model.save_pretrained(self.dygraph_model_path)
        tokenizer.save_pretrained(self.dygraph_model_path)

        # export to static
        self.static_model_path = os.path.join(self.temp_dir.name, "static")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64",
                                    name='input_ids'),
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64",
                                    name='token_type_ids')
        ]
        static_model = paddle.jit.to_static(model, input_spec=input_spec)
        paddle.jit.save(static_model, self.static_model_path)
        tokenizer.save_pretrained(self.static_model_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    @parameterized.expand([(1, ), (2, )])
    def test_text_classification_taskf(self, batch_size):
        # input_text is a tuple to simulate the args passed from Taskflow to TextClassificationTask
        input_text = (["百度", "深度学习框架", "飞桨", "PaddleNLP"], )
        id2label = {
            0: "negative",
            1: "positive",
        }
        dygraph_taskflow = TextClassificationTask(
            model="multi_class",
            task="text_classification",
            task_path=self.dygraph_model_path,
            id2label=id2label,
            batch_size=batch_size)

        dygraph_results = dygraph_taskflow(input_text)
        self.assertEqual(len(dygraph_results), len(input_text[0]))

        static_taskflow = TextClassificationTask(
            model="multi_class",
            task="text_classification",
            is_static_model=True,
            task_path=self.static_model_path,
            id2label=id2label,
            batch_size=batch_size)

        static_results = static_taskflow(input_text)
        self.assertEqual(len(static_results), len(input_text[0]))

        for dygraph_result, static_result in zip(dygraph_results,
                                                 static_results):
            self.assertEqual(dygraph_result["label"], static_result["label"])
            self.assertAlmostEqual(dygraph_result["score"],
                                   static_result["score"],
                                   delta=1e-6)

    @parameterized.expand([(1, ), (2, )])
    def test_taskflow(self, batch_size):
        input_text = ["百度", "深度学习框架", "飞桨", "PaddleNLP"]
        id2label = {
            0: "negative",
            1: "positive",
        }
        dygraph_taskflow = Taskflow(model="multi_class",
                                    task="text_classification",
                                    task_path=self.dygraph_model_path,
                                    id2label=id2label,
                                    batch_size=batch_size)

        dygraph_results = dygraph_taskflow(input_text)
        self.assertEqual(len(dygraph_results), len(input_text))

        static_taskflow = Taskflow(model="multi_class",
                                   task="text_classification",
                                   is_static_model=True,
                                   task_path=self.static_model_path,
                                   id2label=id2label,
                                   batch_size=batch_size)

        static_results = static_taskflow(input_text)
        self.assertEqual(len(static_results), len(input_text))

        for dygraph_result, static_result in zip(dygraph_results,
                                                 static_results):
            self.assertEqual(dygraph_result["label"], static_result["label"])
            self.assertAlmostEqual(dygraph_result["score"],
                                   static_result["score"],
                                   delta=1e-6)
