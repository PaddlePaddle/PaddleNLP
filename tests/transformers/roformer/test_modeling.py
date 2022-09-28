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
from typing import Optional, Tuple
from dataclasses import dataclass, fields, Field
from parameterized import parameterized_class

import paddle
from paddle import Tensor

from paddlenlp.transformers import (RoFormerModel, RoFormerPretrainedModel,
                                    RoFormerForSequenceClassification,
                                    RoFormerForTokenClassification,
                                    RoFormerForQuestionAnswering,
                                    RoFormerForMultipleChoice,
                                    RoFormerForMaskedLM)

from ..test_modeling_common import ids_tensor, floats_tensor, random_attention_mask, ModelTesterMixin
from ...testing_utils import slow


@dataclass
class RoFormerModelTestModelConfig:
    """RoFormerModel model config which keep consist with pretrained_init_configuration sub fields
    """
    vocab_size: int = 200
    embedding_size: int = 50
    hidden_size: int = 36
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 16
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 20
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    pad_token_id: int = 0
    rotary_value: bool = False

    @property
    def model_kwargs(self) -> dict:
        """get the model kwargs configuration to init the model"""
        model_config_fields: Tuple[Field,
                                   ...] = fields(RoFormerModelTestModelConfig)
        return {
            field.name: getattr(self, field.name)
            for field in model_config_fields
        }


@dataclass
class RoFormerModelTestConfig(RoFormerModelTestModelConfig):
    """train config under unittest code"""
    batch_size: int = 2
    seq_length: int = 7
    is_training: bool = False
    use_input_mask: bool = False
    use_token_type_ids: bool = True
    type_sequence_label_size = 3

    # used for sequence classification
    num_classes: int = 3
    num_choices: int = 3


class RoFormerModelTester:

    def __init__(self,
                 parent,
                 config: Optional[RoFormerModelTestConfig] = None):
        self.parent = parent
        self.config: RoFormerModelTestConfig = config or RoFormerModelTestConfig(
        )

        self.is_training = self.config.is_training
        self.num_classes = self.config.num_classes
        self.num_choices = self.config.num_choices

        self.type_sequence_label_size = self.config.type_sequence_label_size

    def prepare_config_and_inputs(self):
        config = self.config
        input_ids = ids_tensor([config.batch_size, config.seq_length],
                               config.vocab_size)

        input_mask = None
        if self.config.use_input_mask:
            input_mask = random_attention_mask(
                [config.batch_size, config.seq_length])

        token_type_ids = None
        if self.config.use_token_type_ids:
            token_type_ids = ids_tensor([config.batch_size, config.seq_length],
                                        config.type_vocab_size)
        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size],
                                         self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length],
                                      self.num_classes)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

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
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self) -> dict:
        return self.config.model_kwargs

    def __getattr__(self, key: str):
        if not hasattr(self.config, key):
            raise AttributeError(f'attribute <{key}> not exist')
        return getattr(self.config, key)

    def create_and_check_model(self, config, input_ids: Tensor,
                               token_type_ids: Tensor, input_mask: Tensor,
                               sequence_labels: Tensor, token_labels: Tensor,
                               choice_labels: Tensor):
        model = RoFormerModel(**config)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       return_dict=self.parent.return_dict)
        result = model(input_ids,
                       token_type_ids=token_type_ids,
                       return_dict=self.parent.return_dict)
        result = model(input_ids, return_dict=self.parent.return_dict)

        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.seq_length,
            self.config.hidden_size
        ])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.hidden_size])

    def create_and_check_for_multiple_choice(self, config, input_ids: Tensor,
                                             token_type_ids: Tensor,
                                             input_mask: Tensor,
                                             sequence_labels: Tensor,
                                             token_labels: Tensor,
                                             choice_labels: Tensor):
        model = RoFormerForMultipleChoice(RoFormerModel(**config),
                                          num_choices=self.config.num_choices)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(
            [-1, self.config.num_choices, -1])

        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(1).expand(
                [-1, self.config.num_choices, -1])

        if input_mask is not None:
            input_mask = input_mask.unsqueeze(1).expand(
                [-1, self.config.num_choices, -1])

        result = model(multiple_choice_inputs_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       labels=choice_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if paddle.is_tensor(result):
            result = [result]
        elif choice_labels is not None:
            result = result[1:]

        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.num_choices])

    def create_and_check_for_question_answering(self, config, input_ids: Tensor,
                                                token_type_ids: Tensor,
                                                input_mask: Tensor,
                                                sequence_labels: Tensor,
                                                token_labels: Tensor,
                                                choice_labels: Tensor):
        model = RoFormerForQuestionAnswering(RoFormerModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       start_positions=sequence_labels,
                       end_positions=sequence_labels,
                       return_dict=self.parent.return_dict)

        if paddle.is_tensor(result):
            result = [result]
        elif choice_labels is not None:
            result = result[1:]

        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.seq_length])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.seq_length])

    def create_and_check_for_token_classification(
            self, config, input_ids: Tensor, token_type_ids: Tensor,
            input_mask: Tensor, sequence_labels: Tensor, token_labels: Tensor,
            choice_labels: Tensor):
        model = RoFormerForTokenClassification(RoFormerModel(**config),
                                               num_classes=self.num_classes)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       labels=token_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))
        if paddle.is_tensor(result):
            result = [result]
        elif choice_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.seq_length,
            self.config.num_classes
        ])

    def create_and_check_for_masked_lm(self, config, input_ids: Tensor,
                                       token_type_ids: Tensor,
                                       input_mask: Tensor,
                                       sequence_labels: Tensor,
                                       token_labels: Tensor,
                                       choice_labels: Tensor):
        model = RoFormerForMaskedLM(RoFormerModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       labels=token_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if paddle.is_tensor(result):
            result = [result]
        elif choice_labels is not None:
            result = result[1:]
        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.seq_length,
            self.config.vocab_size
        ])

    def create_and_check_for_sequence_classification(
            self, config, input_ids: Tensor, token_type_ids: Tensor,
            input_mask: Tensor, sequence_labels: Tensor, token_labels: Tensor,
            choice_labels: Tensor):
        model = RoFormerForSequenceClassification(
            RoFormerModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       labels=sequence_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))
        if paddle.is_tensor(result):
            result = [result]
        elif choice_labels is not None:
            result = result[1:]
        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.num_classes])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, token_type_ids, input_mask, _, _,
         _) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask
        }
        return config, inputs_dict


@parameterized_class(("return_dict", "use_labels"), [
    [False, False],
    [False, True],
    [True, False],
    [True, True],
])
class RoFormerModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = RoFormerModel
    use_labels = False
    return_dict = False

    all_model_classes = (RoFormerModel, RoFormerForSequenceClassification,
                         RoFormerForTokenClassification,
                         RoFormerForQuestionAnswering,
                         RoFormerForMultipleChoice, RoFormerForMaskedLM)

    def setUp(self):
        self.model_tester = RoFormerModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(
            *config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(
            *config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(
            *config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(
            *config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(
                RoFormerPretrainedModel.pretrained_init_configuration)[:1]:
            model = RoFormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class RoFormerModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_no_attention(self):
        model = RoFormerModel.from_pretrained("roformer-chinese-small")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 384]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[0.17788891, -2.17795515, 0.28824317],
              [-1.70342600, -2.84062195, -0.53377795],
              [-0.16374627, -0.67967212, -0.37192002]]])

        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = RoFormerModel.from_pretrained("roformer-chinese-small")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 384]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[0.17788891, -2.17795515, 0.28824317],
              [-1.70342600, -2.84062195, -0.53377795],
              [-0.16374627, -0.67967212, -0.37192002]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
