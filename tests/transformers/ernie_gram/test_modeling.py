# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Dict, Any
import paddle
from paddle import Tensor

from dataclasses import dataclass, asdict, fields, Field
from paddlenlp.transformers import (ErnieGramModel, ErnieGramPretrainedModel,
                                    ErnieGramForSequenceClassification,
                                    ErnieGramForTokenClassification,
                                    ErnieGramForQuestionAnswering)

from tests.transformers.test_modeling_common import (ids_tensor, floats_tensor,
                                                     random_attention_mask,
                                                     ModelTesterMixin)
from tests.testing_utils import slow


@dataclass
class ErnieGramTestModelConfig:
    """ernie-gram model config which keep consist with pretrained_init_configuration sub fields
    """
    attention_probs_dropout_prob: float = 0.1
    emb_size: int = 768
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 768
    initializer_range: float = 0.02
    max_position_embeddings: int = 512
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    type_vocab_size: int = 2
    vocab_size: int = 1801

    @property
    def model_kwargs(self) -> dict:
        """get the model kwargs configuration to init the model"""
        model_config_fields: Tuple[Field,
                                   ...] = fields(ErnieGramTestModelConfig)
        return {
            field.name: getattr(self, field.name)
            for field in model_config_fields
        }


@dataclass
class ErnieGramTestConfig(ErnieGramTestModelConfig):
    """all of ErnieGram Test configuration
    
    """
    batch_size: int = 2
    seq_length: int = 7

    is_training: bool = False
    use_token_type_ids: bool = True
    use_attention_mask: bool = True

    # used for sequence classification
    num_classes: int = 3

    test_resize_embeddings: bool = False


class ErnieGramModelTester:
    """Base ErnieGram Model tester which can test:
    """

    def __init__(self, parent, config: Optional[ErnieGramTestConfig] = None):
        self.parent = parent
        self.config: ErnieGramTestConfig = config or ErnieGramTestConfig()

        self.is_training = self.config.is_training

    def __getattr__(self, key: str):
        if not hasattr(self.config, key):
            raise AttributeError(f'attribute <{key}> not exist')
        return getattr(self.config, key)

    def prepare_config_and_inputs(
            self) -> Tuple[Dict[str, Any], Tensor, Tensor, Tensor]:
        config = self.config
        input_ids = ids_tensor([config.batch_size, config.seq_length],
                               config.vocab_size)

        attention_mask = None
        if config.use_attention_mask:
            attention_mask = random_attention_mask(
                [config.batch_size, config.seq_length])

        token_type_ids = None
        if config.use_token_type_ids:
            token_type_ids = paddle.zeros_like(input_ids)

        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size],
                                         self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length],
                                      self.num_classes)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, token_type_ids, attention_mask, sequence_labels, token_labels, choice_labels

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, token_type_ids, attention_mask, _, _, _ = self.prepare_config_and_inputs(
        )
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model(self, config, input_ids: Tensor,
                               token_type_ids: Tensor, attention_mask: Tensor,
                               sequence_labels: Tensor, token_labels: Tensor,
                               choice_labels: Tensor):
        model = ErnieGramModel(**config)
        model.eval()

        result = model(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       return_dict=self.parent.return_dict)
        if paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.seq_length,
            self.config.hidden_size
        ])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.hidden_size])

    def create_and_check_for_sequence_classification(
            self, config, input_ids: Tensor, token_type_ids: Tensor,
            attention_mask: Tensor, sequence_labels: Tensor,
            token_labels: Tensor, choice_labels: Tensor):
        model = ErnieGramForSequenceClassification(
            ErnieGramModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result = model(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       labels=sequence_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.num_classes])

    def create_and_check_for_question_answering(self, config, input_ids: Tensor,
                                                token_type_ids: Tensor,
                                                attention_mask: Tensor,
                                                sequence_labels: Tensor,
                                                token_labels: Tensor,
                                                choice_labels: Tensor):
        model = ErnieGramForQuestionAnswering(ErnieGramModel(**config))
        model.eval()
        result = model(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       start_position=sequence_labels,
                       end_position=sequence_labels,
                       return_dict=self.parent.return_dict)
        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.seq_length])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.seq_length])

    def create_and_check_for_token_classification(
            self, config, input_ids: Tensor, token_type_ids: Tensor,
            attention_mask: Tensor, sequence_labels: Tensor,
            token_labels: Tensor, choice_labels: Tensor):
        model = ErnieGramForTokenClassification(
            ErnieGramModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result = model(input_ids,
                       token_type_ids=token_type_ids,
                       labels=token_labels,
                       return_dict=self.parent.return_dict,
                       attention_mask=attention_mask)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))
        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.seq_length,
            self.config.num_classes
        ])

    def get_config(self) -> dict:
        """get the base model kwargs

        Returns:
            dict: the values of kwargs
        """
        return self.config.model_kwargs


class ErnieGramModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = ErnieGramModel
    return_dict = False
    use_labels = False

    all_model_classes = (ErnieGramModel, ErnieGramForSequenceClassification,
                         ErnieGramForTokenClassification,
                         ErnieGramForQuestionAnswering)

    def setUp(self):
        self.model_tester = ErnieGramModelTester(self)
        self.test_resize_embeddings = self.model_tester.config.test_resize_embeddings

    def get_config():
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(
            *config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(
            *config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(
            *config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(
                ErnieGramPretrainedModel.pretrained_init_configuration)[:1]:
            model = ErnieGramModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class ErnieGramModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_no_attention(self):
        model = ErnieGramModel.from_pretrained("ernie-gram-zh")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[-0.43569842, -1.50805628, -2.24448967],
              [-0.12123521, -1.35024536, -1.76512492],
              [-0.14853711, -1.13618660, -2.87098265]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-5))

    @slow
    def test_inference_with_attention(self):
        model = ErnieGramModel.from_pretrained(
            "ernie-gram-zh-finetuned-dureader-robust")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[0.37543082, -2.94639230, -2.04799986],
              [0.14168003, -2.02873731, -2.34919119],
              [0.70280838, -2.40280604, -1.93488157]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
