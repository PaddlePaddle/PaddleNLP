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

import unittest
from tempfile import TemporaryDirectory

from paddlenlp.taskflow import Taskflow
from paddlenlp.taskflow.text_feature_extraction import (
    SentenceFeatureExtractionTask,
    TextFeatureExtractionTask,
)


class TestTextFeatureExtractionTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = TemporaryDirectory()
        cls.max_seq_len = 32
        cls.model = "__internal_testing__/tiny-random-rocketqa-query-encoder"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @unittest.skipIf(True, "TODO, fix ci for new from_pretrained!")
    def test_text_feature_extraction_task(self):
        input_text = (["这是一只猫", "这是一只狗"],)
        # dygraph text test
        dygraph_taskflow = TextFeatureExtractionTask(
            model="rocketqa-zh-nano-query-encoder",
            task="feature_extraction",
            task_path=self.model,
            _static_mode=False,
            device_id=0,
        )
        dygraph_results = dygraph_taskflow(input_text)
        shape = dygraph_results["features"].shape
        self.assertEqual(shape[0], 2)

        # static text test
        static_taskflow = TextFeatureExtractionTask(
            model="rocketqa-zh-nano-query-encoder",
            task="feature_extraction",
            task_path=self.model,
            _static_mode=True,
            device_id=0,
        )
        static_results = static_taskflow(input_text)
        shape = static_results["features"].shape
        self.assertEqual(shape[0], 2)

        for dygraph_result, static_result in zip(dygraph_results["features"], static_results["features"]):
            for dygraph_pred, static_pred in zip(dygraph_result.tolist(), static_result.tolist()):
                self.assertAlmostEqual(dygraph_pred, static_pred, delta=1e-5)

    @unittest.skipIf(True, "TODO, fix ci for new from_pretrained!")
    def test_taskflow_task(self):
        input_text = ["这是一只猫", "这是一只狗"]
        # dygraph test
        dygraph_taskflow = Taskflow(
            model="rocketqa-zh-nano-query-encoder",
            task="feature_extraction",
            task_path=self.model,
            _static_mode=False,
        )
        dygraph_results = dygraph_taskflow(input_text)
        shape = dygraph_results["features"].shape

        self.assertEqual(shape[0], 2)

        # static test
        static_taskflow = Taskflow(
            model="rocketqa-zh-nano-query-encoder",
            task="feature_extraction",
            task_path=self.model,
            _static_mode=True,
        )
        static_results = static_taskflow(input_text)
        self.assertEqual(static_results["features"].shape[0], 2)

        for dygraph_result, static_result in zip(dygraph_results["features"], static_results["features"]):
            for dygraph_pred, static_pred in zip(dygraph_result.tolist(), static_result.tolist()):
                self.assertAlmostEqual(dygraph_pred, static_pred, delta=1e-5)


class TestSentenceeExtractionTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = TemporaryDirectory()
        cls.max_seq_len = 32
        cls.model = "__internal_testing__/tiny-random-m3e"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_text_feature_extraction_task(self):
        input_text = (["这是一只猫", "这是一只狗"],)
        # dygraph text test
        dygraph_taskflow = SentenceFeatureExtractionTask(
            model=self.model,
            task="feature_extraction",
            _static_mode=False,
            device_id=0,
        )
        dygraph_results = dygraph_taskflow(input_text)
        shape = dygraph_results["features"].shape
        self.assertEqual(shape, [2, 768])

        # static text test
        static_taskflow = SentenceFeatureExtractionTask(
            model=self.model,
            task="feature_extraction",
            _static_mode=True,
            device_id=0,
        )
        static_results = static_taskflow(input_text)
        shape = static_results["features"].shape
        self.assertEqual(shape, [2, 768])
        for dygraph_result, static_result in zip(dygraph_results["features"], static_results["features"]):
            for dygraph_pred, static_pred in zip(dygraph_result.tolist(), static_result.tolist()):
                self.assertAlmostEqual(dygraph_pred, static_pred, delta=1e-5)

    def test_taskflow_task(self):
        input_text = ["这是一只猫", "这是一只狗"]
        # dygraph test
        dygraph_taskflow = Taskflow(
            model=self.model,
            task="feature_extraction",
            _static_mode=False,
        )
        dygraph_results = dygraph_taskflow(input_text)
        shape = dygraph_results["features"].shape

        self.assertEqual(shape, [2, 768])
        # static test
        static_taskflow = Taskflow(
            model=self.model,
            task="feature_extraction",
            _static_mode=True,
        )
        static_results = static_taskflow(input_text)
        self.assertEqual(static_results["features"].shape, [2, 768])

        for dygraph_result, static_result in zip(dygraph_results["features"], static_results["features"]):
            for dygraph_pred, static_pred in zip(dygraph_result.tolist(), static_result.tolist()):
                self.assertAlmostEqual(dygraph_pred, static_pred, delta=1e-5)
