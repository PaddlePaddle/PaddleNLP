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
import unittest

import numpy as np

from paddlenlp.datasets import InTokensIterableDataset, InTokensMapDataset, load_dataset
from paddlenlp.transformers import AutoTokenizer
from tests.testing_utils import get_tests_dir


class TestInTokensMapDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))
        cls.train_ds = load_dataset(
            "clue",
            "tnews",
            data_files=[os.path.join(fixture_path, "tnews", "train.json")],
            lazy=False,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/micro-random-llama")
        cls.tokenizer.pad_token_id = 0
        cls.dataset = cls.train_ds.map(lambda example: cls.preprocess_function(cls, example))

    def preprocess_function(self, example, max_src_length=3, max_tgt_length=3):
        inputs = example["sentence"][:2]
        model_inputs = self.tokenizer(inputs, max_length=max_src_length, truncation=True, return_attention_mask=False)
        labels_input_ids = model_inputs["input_ids"] + [self.tokenizer.eos_token_id]
        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + labels_input_ids
        model_inputs["input_ids"] = model_inputs["input_ids"] + labels_input_ids
        model_inputs["position_ids"] = list(range(len(model_inputs["input_ids"])))
        model_inputs["attention_mask"] = np.tril(
            np.ones([len(model_inputs["input_ids"]), len(model_inputs["input_ids"])]), 0
        ).tolist()
        return model_inputs

    def test_map_dataset(
        self,
    ):

        inData = InTokensMapDataset(self.dataset, self.tokenizer, max_length=128)
        # Test shape
        self.assertEqual(list(inData[0].keys()), ["input_ids", "labels", "position_ids", "attention_mask"])
        self.assertEqual(len(inData), 1)
        self.assertEqual(type(inData[0]["input_ids"]), list)
        self.assertEqual(np.array(inData[0]["input_ids"]).shape, (70,))

        # Test intokens
        inData = InTokensMapDataset(self.dataset, self.tokenizer, max_length=16)
        expected_output = {
            "input_ids": [1, 29871, 30429, 1, 29871, 30429, 2, 1, 29871, 31427, 1, 29871, 31427, 2],
            "labels": [-100, -100, -100, 1, 29871, 30429, 2, -100, -100, -100, 1, 29871, 31427, 2],
            "position_ids": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
            "attention_mask": [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            ],
        }
        self.assertEqual(inData[0]["input_ids"], expected_output["input_ids"])
        self.assertEqual(inData[0]["position_ids"], expected_output["position_ids"])
        self.assertEqual(inData[0]["labels"], expected_output["labels"])
        self.assertEqual(inData[0]["attention_mask"], expected_output["attention_mask"])

    def test_missing_data(self):
        orginal_input_ids = [item["input_ids"] for item in self.dataset]
        orginal_input_ids = [sum(orginal_input_ids, [])]
        inData = InTokensMapDataset(self.dataset, self.tokenizer, max_length=128)
        tgt_input_ids = [item["input_ids"] for item in inData]
        self.assertEqual(orginal_input_ids, tgt_input_ids)


class TestInTokensIterableDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_path = get_tests_dir(os.path.join("fixtures", "dummy"))
        cls.train_ds = load_dataset(
            "clue",
            "tnews",
            data_files=[os.path.join(fixture_path, "tnews", "train.json")],
            lazy=False,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/micro-random-llama")
        cls.tokenizer.pad_token_id = 0
        cls.dataset = cls.train_ds.map(lambda example: cls.preprocess_function(cls, example))

    def preprocess_function(self, example, max_src_length=3, max_tgt_length=3):
        inputs = example["sentence"][:2]
        model_inputs = self.tokenizer(inputs, max_length=max_src_length, truncation=True, return_attention_mask=True)
        labels_input_ids = model_inputs["input_ids"] + [self.tokenizer.eos_token_id]
        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + labels_input_ids
        model_inputs["input_ids"] = model_inputs["input_ids"] + labels_input_ids
        model_inputs["position_ids"] = list(range(len(model_inputs["input_ids"])))
        model_inputs["attention_mask"] = np.tril(
            np.ones([len(model_inputs["input_ids"]), len(model_inputs["input_ids"])]), 0
        ).tolist()
        return model_inputs

    def test_iterable_dataset(self):
        inData = InTokensIterableDataset(self.dataset, self.tokenizer, max_length=128)

        example = []
        for item in inData:
            example.append(item)
            break

        # Test shape
        self.assertEqual(list(example[0].keys()), ["input_ids", "attention_mask", "labels", "position_ids"])
        self.assertEqual(type(example[0]["input_ids"]), list)
        self.assertEqual(np.array(example[0]["input_ids"]).shape, (70,))

        inData = InTokensIterableDataset(self.dataset, self.tokenizer, max_length=16)

        expected_output = {
            "input_ids": [1, 29871, 30429, 1, 29871, 30429, 2, 1, 29871, 31427, 1, 29871, 31427, 2],
            "labels": [-100, -100, -100, 1, 29871, 30429, 2, -100, -100, -100, 1, 29871, 31427, 2],
            "position_ids": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
            "attention_mask": [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            ],
        }
        # Get one data
        example = []
        for item in inData:
            example.append(item)
            break
        self.assertEqual(example[0]["input_ids"], expected_output["input_ids"])
        self.assertEqual(example[0]["position_ids"], expected_output["position_ids"])
        self.assertEqual(example[0]["labels"], expected_output["labels"])
        self.assertEqual(example[0]["attention_mask"], expected_output["attention_mask"])

    def test_missing_data(self):
        orginal_input_ids = [item["input_ids"] for item in self.dataset]
        orginal_input_ids = [sum(orginal_input_ids, [])]
        inData = InTokensIterableDataset(self.dataset, self.tokenizer, max_length=128)
        tgt_input_ids = [item["input_ids"] for item in inData]
        self.assertEqual(orginal_input_ids, tgt_input_ids)
