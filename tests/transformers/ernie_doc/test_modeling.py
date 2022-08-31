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

from parameterized import parameterized_class
import paddle
from paddle import Tensor

from dataclasses import dataclass, asdict, fields, Field
from paddlenlp.transformers import (
    ErnieDocModel,
    ErnieDocPretrainedModel,
    ErnieDocForSequenceClassification,
    ErnieDocForTokenClassification,
    ErnieDocForQuestionAnswering,
)
from ..test_modeling_common import (ids_tensor, floats_tensor,
                                    random_attention_mask, ModelTesterMixin)
from ...testing_utils import slow


@dataclass
class ErnieDocTestModelConfig:
    """ernie-doc model config which keep consist with pretrained_init_configuration sub fields
    """
    attention_dropout_prob: float = 0.0
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    relu_dropout: float = 0.0
    hidden_size: int = 64
    initializer_range: float = 0.02
    max_position_embeddings: int = 60
    num_attention_heads: int = 8
    num_hidden_layers: int = 8
    task_type_vocab_size: int = 3
    vocab_size: int = 100
    memory_len: int = 50
    epsilon: float = 1e-12
    pad_token_id: int = 1

    @property
    def model_kwargs(self) -> dict:
        """get the model kwargs configuration to init the model"""
        model_config_fields: Tuple[Field, ...] = fields(ErnieDocTestModelConfig)
        return {
            field.name: getattr(self, field.name)
            for field in model_config_fields
        }


@dataclass
class ErnieDocTestConfig(ErnieDocTestModelConfig):
    """all of ErnieDoc Test configuration
    """
    batch_size: int = 2
    is_training: bool = False
    use_input_mask: bool = False
    use_token_type_ids: bool = False
    seq_length: int = 50

    # seq_length + memeory_length
    key_length: int = 100

    # used for sequence classification
    type_sequence_label_size: int = 3
    num_classes: int = 3


class ErnieDocModelTester:
    """Base ErnieDoc Model tester which can test:
    """

    def __init__(self, parent, config: Optional[ErnieDocTestConfig] = None):
        self.parent = parent
        self.config = config or ErnieDocTestConfig()

        self.is_training = self.config.is_training

    def prepare_config_and_inputs(
            self) -> Tuple[Dict[str, Any], Tensor, Tensor, Tensor]:
        config = self.config
        input_ids = ids_tensor([config.batch_size, config.memory_len],
                               config.vocab_size)

        input_mask = None
        if config.use_input_mask:
            input_mask = random_attention_mask(
                [config.batch_size, config.memory_len])

        token_type_ids = None
        if config.use_token_type_ids:
            token_type_ids = ids_tensor([config.batch_size, config.memory_len],
                                        config.type_vocab_size)

        sequence_labels = None
        token_labels = None
        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size],
                                         self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length],
                                      self.num_classes)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels

    def __getattr__(self, key: str):
        if not hasattr(self.config, key):
            raise AttributeError(f'attribute <{key}> not exist')
        return getattr(self.config, key)

    def create_and_check_model(self, config, input_ids, token_type_ids,
                               input_mask, sequence_labels, token_labels):
        model = ErnieDocModel(**config)
        model.eval()

        result = model(input_ids,
                       attn_mask=input_mask,
                       token_type_ids=token_type_ids,
                       return_dict=self.parent.return_dict)
        result = model(input_ids,
                       token_type_ids=token_type_ids,
                       return_dict=self.parent.return_dict)
        result = model(input_ids, return_dict=self.parent.return_dict)

        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.memory_len,
            self.config.hidden_size
        ])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.hidden_size])

    def create_and_check_for_sequence_classification(self, config, input_ids,
                                                     token_type_ids, input_mask,
                                                     sequence_labels,
                                                     token_labels):
        model = ErnieDocForSequenceClassification(
            ErnieDocModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result = model(input_ids,
                       attn_mask=input_mask,
                       token_type_ids=token_type_ids,
                       labels=sequence_labels,
                       return_dict=self.parent.return_dict)
        if token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.num_classes])

    def create_and_check_for_token_classification(self, config, input_ids,
                                                  token_type_ids, input_mask,
                                                  sequence_labels,
                                                  token_labels):
        model = ErnieDocForTokenClassification(
            ErnieDocModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result = model(input_ids,
                       attn_mask=input_mask,
                       token_type_ids=token_type_ids,
                       labels=token_labels,
                       return_dict=self.parent.return_dict)
        if token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.memory_len,
            self.config.num_classes
        ])

    def create_and_check_for_question_answering(self, config, input_ids,
                                                token_type_ids, input_mask,
                                                sequence_labels, token_labels):
        model = ErnieDocForQuestionAnswering(ErnieDocModel(**config))
        model.eval()

        result = model(input_ids,
                       position_ids=None,
                       attn_mask=input_mask,
                       token_type_ids=token_type_ids,
                       start_positions=sequence_labels,
                       end_positions=sequence_labels,
                       return_dict=self.parent.return_dict)

        if sequence_labels is not None:
            result = result[1:]

        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.memory_len])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.memory_len])

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, token_type_ids, input_mask, _, _ = self.prepare_config_and_inputs(
        )
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attn_mask": input_mask,
        }
        return config, inputs_dict

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
class ErnieDocModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = ErnieDocModel

    use_labels = False
    return_dict = False

    all_model_classes = (
        ErnieDocModel,
        ErnieDocForSequenceClassification,
        ErnieDocForTokenClassification,
        ErnieDocForQuestionAnswering,
    )

    def setUp(self):
        self.model_tester = ErnieDocModelTester(self)

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
        self.model_tester.create_and_check_for_question_answering(
            *config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(
                ErnieDocPretrainedModel.pretrained_init_configuration)[:1]:
            model = ErnieDocModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class ErnieDocModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_no_attention(self):
        model = ErnieDocModel.from_pretrained("ernie-doc-base-en")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[-0.17067151, -0.12679860, 0.10108676],
              [-0.15048309, -0.11452073, 0.27110466],
              [-0.64834023, 0.05063335, 0.07601062]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-5))

    @slow
    def test_inference_with_attention(self):
        model = ErnieDocModel.from_pretrained("ernie-doc-base-en")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attn_mask = paddle.ones(shape=[1, 11, 1])

        with paddle.no_grad():
            output = model(input_ids, attn_mask=attn_mask)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[-0.17067151, -0.12679860, 0.10108676],
              [-0.15048309, -0.11452073, 0.27110466],
              [-0.64834023, 0.05063335, 0.07601062]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
