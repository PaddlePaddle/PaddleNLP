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

import paddle
from parameterized import parameterized

from paddlenlp.taskflow import Taskflow
from paddlenlp.taskflow.text_classification import TextClassificationTask
from paddlenlp.transformers import AutoTokenizer, ErnieForSequenceClassification


class TestTextClassificationTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = TemporaryDirectory()
        cls.dygraph_model_path = os.path.join(cls.temp_dir.name, "dygraph")
        model = ErnieForSequenceClassification.from_pretrained("__internal_testing__/ernie", num_classes=2)
        tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/ernie")
        model.save_pretrained(cls.dygraph_model_path)
        tokenizer.save_pretrained(cls.dygraph_model_path)

        # export to static
        cls.static_model_path = os.path.join(cls.temp_dir.name, "static")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
        ]
        static_model = paddle.jit.to_static(model, input_spec=input_spec)
        paddle.jit.save(static_model, cls.static_model_path)
        tokenizer.save_pretrained(cls.static_model_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @parameterized.expand(
        [
            (1, "multi_class"),
            (2, "multi_class"),
            (1, "multi_label"),
            (2, "multi_label"),
        ]
    )
    def test_classification_task(self, batch_size, model):
        # input_text is a tuple to simulate the args passed from Taskflow to TextClassificationTask
        input_text = (["百度", "深度学习框架", "飞桨", "PaddleNLP"],)
        id2label = {
            0: "negative",
            1: "positive",
        }
        dygraph_taskflow = TextClassificationTask(
            model=model,
            task="text_classification",
            task_path=self.dygraph_model_path,
            id2label=id2label,
            batch_size=batch_size,
            device_id=0,
        )

        dygraph_results = dygraph_taskflow(input_text)

        self.assertEqual(len(dygraph_results), len(input_text[0]))

        static_taskflow = TextClassificationTask(
            model=model,
            task="text_classification",
            is_static_model=True,
            task_path=self.static_model_path,
            id2label=id2label,
            batch_size=batch_size,
            device_id=0,
        )

        static_results = static_taskflow(input_text)
        self.assertEqual(len(static_results), len(input_text[0]))

        for dygraph_result, static_result in zip(dygraph_results, static_results):
            for dygraph_pred, static_pred in zip(dygraph_result["predictions"], static_result["predictions"]):
                self.assertEqual(dygraph_pred["label"], static_pred["label"])
                self.assertAlmostEqual(dygraph_pred["score"], static_pred["score"], delta=1e-6)
                # if multi_label, all predictions should be greater than the threshold
                if model == "multi_label":
                    self.assertGreater(dygraph_pred["score"], dygraph_taskflow.multilabel_threshold)

    @parameterized.expand(
        [
            (1, "multi_class"),
            (2, "multi_class"),
            (1, "multi_label"),
            (2, "multi_label"),
        ]
    )
    def test_taskflow(self, batch_size, model):
        input_text = ["百度", "深度学习框架", "飞桨", "PaddleNLP"]
        id2label = {
            0: "negative",
            1: "positive",
        }
        dygraph_taskflow = Taskflow(
            model=model,
            task="text_classification",
            task_path=self.dygraph_model_path,
            id2label=id2label,
            batch_size=batch_size,
        )

        dygraph_results = dygraph_taskflow(input_text)
        self.assertEqual(len(dygraph_results), len(input_text))

        static_taskflow = Taskflow(
            model=model,
            task="text_classification",
            is_static_model=True,
            task_path=self.static_model_path,
            id2label=id2label,
            batch_size=batch_size,
        )

        static_results = static_taskflow(input_text)
        self.assertEqual(len(static_results), len(input_text))

        for dygraph_result, static_result in zip(dygraph_results, static_results):
            for dygraph_pred, static_pred in zip(dygraph_result["predictions"], static_result["predictions"]):
                self.assertEqual(dygraph_pred["label"], static_pred["label"])
                self.assertAlmostEqual(dygraph_pred["score"], static_pred["score"], delta=1e-6)
                # if multi_label, all predictions should be greater than the threshold
                if model == "multi_label":
                    self.assertGreater(dygraph_pred["score"], dygraph_taskflow.task_instance.multilabel_threshold)
