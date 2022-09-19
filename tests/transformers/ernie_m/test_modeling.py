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
from parameterized import parameterized_class

from dataclasses import dataclass, asdict, fields, Field
from paddlenlp.transformers import (ErnieMPretrainedModel, ErnieMModel,
                                    ErnieMForSequenceClassification,
                                    ErnieMForTokenClassification,
                                    ErnieMForQuestionAnswering,
                                    ErnieMForMultipleChoice)
from tests.transformers.test_modeling_common import (ids_tensor, floats_tensor,
                                                     random_attention_mask,
                                                     ModelTesterMixin)
from tests.testing_utils import slow


@dataclass
class ErnieMTestModelConfig:
    """skep model config which keep consist with pretrained_init_configuration sub fields
    """
    attention_probs_dropout_prob: float = 0.1
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 48
    initializer_range: float = 0.02
    max_position_embeddings: int = 20
    num_attention_heads: int = 16
    num_hidden_layers: int = 3
    vocab_size: int = 100
    pad_token_id: int = 1

    @property
    def model_kwargs(self) -> dict:
        """get the model kwargs configuration to init the model"""
        model_config_fields: Tuple[Field, ...] = fields(ErnieMTestModelConfig)
        return {
            field.name: getattr(self, field.name)
            for field in model_config_fields
        }


@dataclass
class ErnieMTestConfig(ErnieMTestModelConfig):
    """all of ErnieM Test configuration
    
    """
    batch_size: int = 2
    seq_length: int = 7

    is_training: bool = False
    use_position_ids: bool = True
    use_attention_mask: bool = True

    type_sequence_label_size: int = 3
    # used for sequence classification
    num_classes: int = 3

    # used for multi-choices
    num_choices: int = 3

    test_resize_embeddings: bool = False


class ErnieMModelTester:
    """Base ErnieM Model tester which can test:
    """

    def __init__(self, parent, config: Optional[ErnieMTestConfig] = None):
        self.parent = parent
        self.config: ErnieMTestConfig = config or ErnieMTestConfig()

        self.is_training = self.config.is_training

        # set multi_choice
        self.num_choices = self.config.num_choices

    def __getattr__(self, key: str):
        if not hasattr(self.config, key):
            raise AttributeError(f'attribute <{key}> not exist')
        return getattr(self.config, key)

    def prepare_config_and_inputs(self):
        config = self.config
        input_ids = ids_tensor([config.batch_size, config.seq_length],
                               config.vocab_size)

        attention_mask = None
        if config.use_attention_mask:
            attention_mask = random_attention_mask(
                [config.batch_size, config.seq_length])

        position_ids = None
        if config.use_position_ids:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones

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
        return config, input_ids, position_ids, attention_mask, sequence_labels, token_labels, choice_labels

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, position_ids, attention_mask, _, _, _ = self.prepare_config_and_inputs(
        )
        inputs_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model(self, config: Dict[str, Any], input_ids: Tensor,
                               position_ids: Tensor, attention_mask: Tensor,
                               sequence_labels: Tensor, token_labels: Tensor,
                               choice_labels: Tensor):
        model = ErnieMModel(**config)
        model.eval()

        result = model(input_ids,
                       attention_mask=attention_mask,
                       position_ids=position_ids,
                       return_dict=self.parent.return_dict)
        result = model(input_ids,
                       position_ids=position_ids,
                       return_dict=self.parent.return_dict)
        result = model(input_ids,
                       attention_mask=attention_mask,
                       return_dict=self.parent.return_dict)

        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.seq_length,
            self.config.hidden_size
        ])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.hidden_size])

    def create_and_check_for_sequence_classification(
            self, config, input_ids: Tensor, position_ids: Tensor,
            attention_mask: Tensor, sequence_labels: Tensor,
            token_labels: Tensor, choice_labels: Tensor):
        model = ErnieMForSequenceClassification(
            ErnieMModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result = model(input_ids,
                       position_ids=position_ids,
                       attention_mask=attention_mask,
                       labels=sequence_labels,
                       return_dict=self.parent.return_dict)

        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.num_classes])

    def create_and_check_for_question_answering(self, config, input_ids: Tensor,
                                                position_ids: Tensor,
                                                attention_mask: Tensor,
                                                sequence_labels: Tensor,
                                                token_labels: Tensor,
                                                choice_labels: Tensor):
        model = ErnieMForQuestionAnswering(ErnieMModel(**config))
        model.eval()
        result = model(input_ids,
                       position_ids=position_ids,
                       attention_mask=attention_mask,
                       start_positions=sequence_labels,
                       end_positions=sequence_labels,
                       return_dict=self.parent.return_dict)

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.seq_length,
            self.config.num_classes
        ])

    def create_and_check_for_token_classification(
            self, config, input_ids: Tensor, position_ids: Tensor,
            attention_mask: Tensor, sequence_labels: Tensor,
            token_labels: Tensor, choice_labels: Tensor):
        model = ErnieMForTokenClassification(
            ErnieMModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result = model(input_ids,
                       attention_mask=attention_mask,
                       position_ids=position_ids,
                       labels=token_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.seq_length,
            self.config.num_classes
        ])

    def create_and_check_for_multiple_choice(self, config, input_ids: Tensor,
                                             position_ids: Tensor,
                                             attention_mask: Tensor,
                                             sequence_labels: Tensor,
                                             token_labels: Tensor,
                                             choice_labels: Tensor):
        model = ErnieMForMultipleChoice(ErnieMModel(**config),
                                        num_choices=self.config.num_choices)
        model.eval()

        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(
            [-1, self.config.num_choices, -1])
        multiple_choice_position_ids = position_ids.unsqueeze(1).expand(
            [-1, self.config.num_choices, -1])
        multiple_choice_attention_mask = attention_mask.unsqueeze(1).expand(
            [-1, self.config.num_choices, -1])

        result = model(multiple_choice_inputs_ids,
                       position_ids=multiple_choice_position_ids,
                       attention_mask=multiple_choice_attention_mask,
                       labels=choice_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))
        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.num_choices])

    def get_config(self) -> dict:
        """get the base model kwargs

        Returns:
            dict: the values of kwargs
        """
        return self.config.model_kwargs


@parameterized_class(("return_dict", "use_labels"), [
    [False, False],
    [False, True],
    [True, False],
    [True, True],
])
class ErnieMModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = ErnieMModel
    use_labels = False
    return_dict = False

    all_model_classes = (ErnieMModel, ErnieMForSequenceClassification,
                         ErnieMForTokenClassification,
                         ErnieMForQuestionAnswering, ErnieMForMultipleChoice)

    def setUp(self):
        self.model_tester = ErnieMModelTester(self)

        # set attribute in setUp to overwrite the static attribute
        self.test_resize_embeddings = False

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

    def test_for_multi_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(
            *config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(
                ErnieMPretrainedModel.pretrained_init_configuration)[:1]:
            model = ErnieMModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class ErnieMModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_no_attention(self):
        model = ErnieMModel.from_pretrained("ernie-m-base")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[-0.02920425, -0.00768885, -0.10219190],
              [-0.10798159, 0.02311476, -0.17285497],
              [0.05675533, 0.01330730, -0.06826267]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-5))

    @slow
    def test_inference_with_attention(self):
        model = ErnieMModel.from_pretrained("ernie-m-base")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[-0.02920425, -0.00768885, -0.10219190],
              [-0.10798159, 0.02311476, -0.17285497],
              [0.05675533, 0.01330730, -0.06826267]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
