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

import paddle

from paddlenlp.transformers import (TinyBertModel, TinyBertForQuestionAnswering,
                                    TinyBertForSequenceClassification,
                                    TinyBertForPretraining,
                                    TinyBertForMultipleChoice,
                                    TinyBertPretrainedModel)
from ..test_modeling_common import ids_tensor, floats_tensor, random_attention_mask, ModelTesterMixin
from ...testing_utils import slow


@dataclass
class TinyBertTestModelConfig:
    """tinybert model config which keep consist with pretrained_init_configuration sub fields
    """
    vocab_size: int = 100
    hidden_size: int = 100
    num_hidden_layers: int = 4
    num_attention_heads: int = 5
    intermediate_size: int = 120
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 62
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    pad_token_id: int = 0

    @property
    def model_kwargs(self) -> dict:
        """get the model kwargs configuration to init the model"""
        model_config_fields: Tuple[Field, ...] = fields(TinyBertTestModelConfig)
        return {
            field.name: getattr(self, field.name)
            for field in model_config_fields
        }


@dataclass
class TinyBertTestConfig(TinyBertTestModelConfig):
    """train config under unittest code"""
    batch_size: int = 2
    seq_length: int = 7
    is_training: bool = False
    use_input_mask: bool = False
    use_token_type_ids: bool = True

    # used for sequence classification
    num_classes: int = 3
    num_choices: int = 3


class TinyBertModelTester:

    def __init__(
        self,
        parent,
        config: Optional[TinyBertTestConfig] = None,
    ):
        self.parent = parent
        self.config: TinyBertTestConfig = config or TinyBertTestConfig()

        self.is_training = self.config.is_training
        self.num_classes = self.config.num_classes
        self.num_choices = self.config.num_choices

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

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask

    def get_config(self) -> dict:
        return self.config.model_kwargs

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = TinyBertModel(**config)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [
            self.config.batch_size, self.config.seq_length,
            self.config.hidden_size
        ])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.hidden_size])

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = TinyBertForMultipleChoice(TinyBertModel(**config),
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

        result = model(
            multiple_choice_inputs_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(
            result.shape, [self.config.batch_size, self.config.num_choices])

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = TinyBertForMaskedLM(TinyBertModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids)
        self.parent.assertEqual(
            result.shape,
            [self.config.batch_size, self.config.seq_length, self.vocab_size])

    def create_and_check_for_question_answering(self, config, input_ids,
                                                token_type_ids, input_mask):
        model = TinyBertForQuestionAnswering(TinyBertModel(**config))
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(
            result[0].shape, [self.config.batch_size, self.config.seq_length])
        self.parent.assertEqual(
            result[1].shape, [self.config.batch_size, self.config.seq_length])

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = TinyBertForSequenceClassification(
            TinyBertModel(**config), num_classes=self.config.num_classes)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(
            result.shape, [self.config.batch_size, self.config.num_classes])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask
        }
        return config, inputs_dict


class TinyBertModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = TinyBertModel

    all_model_classes = (
        TinyBertModel,
        TinyBertForMultipleChoice,
        TinyBertForPretraining,
        TinyBertForQuestionAnswering,
        TinyBertForSequenceClassification,
    )

    def setUp(self):
        self.model_tester = TinyBertModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

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

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(
                TinyBertPretrainedModel.pretrained_init_configuration)[:1]:
            model = TinyBertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_hidden_states_output(self):
        self.skipTest(
            "skip: test_hidden_states_output -> there is no supporting argument return_dict"
        )


class TinyBertModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_no_attention(self):
        model = TinyBertModel.from_pretrained("tinybert-4l-312d")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 312]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[-0.76857519, -0.04066351, -0.36538580],
              [-0.79803109, -0.04977923, -0.37076530],
              [-0.76121056, -0.07496471, -0.35906711]]])

        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = TinyBertModel.from_pretrained("tinybert-4l-312d")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 312]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[-0.76857519, -0.04066351, -0.36538580],
              [-0.79803109, -0.04977923, -0.37076530],
              [-0.76121056, -0.07496471, -0.35906711]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
