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
from tempfile import TemporaryDirectory

from paddlenlp.taskflow import Taskflow
from paddlenlp.taskflow.text_similarity import TextSimilarityTask


class TestTextSimilarityTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = TemporaryDirectory()
        cls.max_seq_len = 32
        cls.model = "__internal_testing__/tiny-random-rocketqa-cross-encoder"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_bert_model(self):
        # static simbert test
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

    def test_text_similarity_task(self):
        # static rocketqa test
        input_text = ([["世界上什么东西最小", "世界上什么东西最小？"]],)
        static_taskflow = TextSimilarityTask(
            model="rocketqa-zh-dureader-cross-encoder",
            task="text_similarity",
            task_path=self.model,
            max_seq_len=self.max_seq_len,
            device_id=0,
        )
        static_results = static_taskflow(input_text)
        self.assertTrue(len(static_results) == 1)
        self.assertTrue("text1" in static_results[0])
        self.assertTrue("text2" in static_results[0])
        self.assertIsInstance(static_results[0]["similarity"], float)

        input_text = ([["光眼睛大就好看吗", "眼睛好看吗？"], ["小蝌蚪找妈妈怎么样", "小蝌蚪找妈妈是谁画的"]],)
        results = static_taskflow(input_text)
        self.assertTrue(len(results) == 2)
        for result in results:
            self.assertTrue("text1" in result)
            self.assertTrue("text2" in result)
            self.assertIsInstance(result["similarity"], float)

        # static rocketqav2 test
        input_text = ([["Tomorrow is another day", "Today is a sunny day"]],)
        static_taskflow = TextSimilarityTask(
            model="rocketqav2-en-marco-cross-encoder",
            task="text_similarity",
            task_path=self.model,
            max_seq_len=self.max_seq_len,
            device_id=0,
        )
        static_results = static_taskflow(input_text)
        self.assertTrue(len(static_results) == 1)
        self.assertTrue("text1" in static_results[0])
        self.assertTrue("text2" in static_results[0])
        self.assertIsInstance(static_results[0]["similarity"], float)

        input_text = (
            [["Tomorrow is another day", "Today is a sunny day"], ["This is my dream", "This is my father"]],
        )
        results = static_taskflow(input_text)
        self.assertTrue(len(results) == 2)
        for result in results:
            self.assertTrue("text1" in result)
            self.assertTrue("text2" in result)
            self.assertIsInstance(result["similarity"], float)

        # static ernie-search test
        input_text = ([["Tomorrow is another day", "Today is a sunny day"]],)
        static_taskflow = TextSimilarityTask(
            model="ernie-search-large-cross-encoder-marco-en",
            task="text_similarity",
            task_path=self.model,
            max_seq_len=self.max_seq_len,
            device_id=0,
        )
        static_results = static_taskflow(input_text)
        self.assertTrue(len(static_results) == 1)
        self.assertTrue("text1" in static_results[0])
        self.assertTrue("text2" in static_results[0])
        self.assertIsInstance(static_results[0]["similarity"], float)

        input_text = (
            [["Tomorrow is another day", "Today is a sunny day"], ["This is my dream", "This is my father"]],
        )
        results = static_taskflow(input_text)
        self.assertTrue(len(results) == 2)
        for result in results:
            self.assertTrue("text1" in result)
            self.assertTrue("text2" in result)
            self.assertIsInstance(result["similarity"], float)

    def test_taskflow_task(self):
        # static rocketqav1 test
        input_text = [["世界上什么东西最小", "世界上什么东西最小？"]]
        static_taskflow = Taskflow(
            model="rocketqa-zh-dureader-cross-encoder",
            task="text_similarity",
            task_path=self.model,
            max_seq_len=self.max_seq_len,
        )
        static_results = static_taskflow(input_text)
        self.assertTrue(len(static_results) == 1)
        self.assertTrue("text1" in static_results[0])
        self.assertTrue("text2" in static_results[0])
        self.assertIsInstance(static_results[0]["similarity"], float)

        # static rocketqav2 test
        input_text = [["Tomorrow is another day", "Today is a sunny day"]]
        static_taskflow = Taskflow(
            model="rocketqav2-en-marco-cross-encoder",
            task="text_similarity",
            task_path=self.model,
            max_seq_len=self.max_seq_len,
        )
        static_results = static_taskflow(input_text)
        self.assertTrue(len(static_results) == 1)
        self.assertTrue("text1" in static_results[0])
        self.assertTrue("text2" in static_results[0])
        self.assertIsInstance(static_results[0]["similarity"], float)

        input_text = [["Tomorrow is another day", "Today is a sunny day"], ["This is my dream", "This is my father"]]
        results = static_taskflow(input_text)
        self.assertTrue(len(results) == 2)
        for result in results:
            self.assertTrue("text1" in result)
            self.assertTrue("text2" in result)
            self.assertIsInstance(result["similarity"], float)

        # static ernie-search test
        input_text = [["Tomorrow is another day", "Today is a sunny day"]]
        static_taskflow = Taskflow(
            model="ernie-search-large-cross-encoder-marco-en",
            task="text_similarity",
            task_path=self.model,
            max_seq_len=self.max_seq_len,
        )
        static_results = static_taskflow(input_text)
        self.assertTrue(len(static_results) == 1)
        self.assertTrue("text1" in static_results[0])
        self.assertTrue("text2" in static_results[0])
        self.assertIsInstance(static_results[0]["similarity"], float)

        input_text = [["Tomorrow is another day", "Today is a sunny day"], ["This is my dream", "This is my father"]]
        results = static_taskflow(input_text)
        self.assertTrue(len(results) == 2)
        for result in results:
            self.assertTrue("text1" in result)
            self.assertTrue("text2" in result)
            self.assertIsInstance(result["similarity"], float)
