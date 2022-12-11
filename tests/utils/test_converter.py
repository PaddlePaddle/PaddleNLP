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
from unittest import TestCase
from paddle import nn
from tests.testing_utils import slow
from paddlenlp.utils.converter import Converter, StateDictKeysChecker
from paddlenlp.transformers import (
    PretrainedModel,
    BertModel,
    BertConfig,
    BertForTokenClassification,
    BertForSequenceClassification,
)


class TestPretrainedModel(PretrainedModel):
    base_model_prefix = "base"


class BaseModel(TestPretrainedModel):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm([3])
        self.linear = nn.Linear(3, 4)


class DownstreamModel(TestPretrainedModel):
    def __init__(self):
        super().__init__()
        self.base = BaseModel()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 6)


class SecondDownstreamModel(TestPretrainedModel):
    def __init__(self):
        super().__init__()
        self.base = BaseModel()
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, 6)


class TestConverter(unittest.TestCase):
    def setUp(self) -> None:
        self.base_model = BaseModel()

        self.downstream_model = DownstreamModel()

    def test_base_base_checking(self):
        checker = StateDictKeysChecker(self.base_model, Converter.get_model_state_dict(self.base_model, True))
        diff_keys = checker.get_diff_keys()
        self.assertEqual(len(diff_keys), 0)

    def test_base_downstream_checking(self):
        checker = StateDictKeysChecker(self.base_model, Converter.get_model_state_dict(self.downstream_model, True))

        unexpected_keys = checker.get_unexpected_keys()
        self.assertEqual(len(unexpected_keys), 4)

        mismatched_keys = checker.get_mismatched_keys()
        self.assertEqual(len(mismatched_keys), 0)

    def test_downstream_base_checking(self):
        checker = StateDictKeysChecker(self.downstream_model, Converter.get_model_state_dict(self.base_model, True))

        unexpected_keys = checker.get_unexpected_keys()
        self.assertEqual(len(unexpected_keys), 0)

        mismatched_keys = checker.get_mismatched_keys()
        self.assertEqual(len(mismatched_keys), 4)

    def test_downstream_downstream_checking(self):
        checker = StateDictKeysChecker(
            self.downstream_model, Converter.get_model_state_dict(SecondDownstreamModel(), True)
        )

        unexpected_keys = checker.get_unexpected_keys()
        self.assertEqual(len(unexpected_keys), 4)

        mismatched_keys = checker.get_mismatched_keys()
        self.assertEqual(len(mismatched_keys), 4)

    @slow
    def test_bert_case(self):
        config = BertConfig()
        bert_model = BertModel(config)
        bert_for_token_model = BertForTokenClassification(config)

        # base-downstream
        checker = StateDictKeysChecker(bert_model, Converter.get_model_state_dict(bert_for_token_model))

        unexpected_keys = checker.get_unexpected_keys()
        self.assertEqual(len(unexpected_keys), 2)

        mismatched_keys = checker.get_mismatched_keys()
        self.assertEqual(len(mismatched_keys), 0)

        # base-base
        checker = StateDictKeysChecker(bert_model, Converter.get_model_state_dict(bert_model))

        unexpected_keys = checker.get_unexpected_keys()
        self.assertEqual(len(unexpected_keys), 0)

        mismatched_keys = checker.get_mismatched_keys()
        self.assertEqual(len(mismatched_keys), 0)

        # downstream-base
        checker = StateDictKeysChecker(
            bert_for_token_model,
            Converter.get_model_state_dict(bert_model),
        )

        unexpected_keys = checker.get_unexpected_keys()
        self.assertEqual(len(unexpected_keys), 0)

        mismatched_keys = checker.get_mismatched_keys()
        self.assertEqual(len(mismatched_keys), 2)

        # downstream-downstream
        checker = StateDictKeysChecker(
            bert_for_token_model,
            Converter.get_model_state_dict(BertForSequenceClassification(config)),
        )

        unexpected_keys = checker.get_unexpected_keys()
        self.assertEqual(len(unexpected_keys), 0)

        mismatched_keys = checker.get_mismatched_keys()
        self.assertEqual(len(mismatched_keys), 0)

    def test_get_num_layer(self):
        """test `get_num_layer` method"""
        layers = [
            "embeddings_project.weight",
            "embeddings_project.bias",
        ]
        num_layer = Converter.get_num_layer(layers)
        self.assertIsNone(num_layer)

        layers = [
            "embeddings_project.weight",
            "embeddings_project.bias",
            "encoder.layer.0.attention.self.query.weight",
            "encoder.layer.1.attention.self.query.weight",
            "encoder.layer.6.attention.self.query.bias",
            "encoder.layer.11.attention.output.LayerNorm.bias",
            "encoder.layer.11.intermediate.dense.bias",
        ]
        num_layer = Converter.get_num_layer(layers)
        self.assertEqual(num_layer, 12)

    def test_remove_unused_fields(self):
        config = {"transformers_version": "1"}
        Converter.remove_transformer_unused_fields(config)
        self.assertNotIn("transformers_version", config)

        # remove un-exist field
        Converter.remove_transformer_unused_fields(config)
        self.assertNotIn("transformers_version", config)
