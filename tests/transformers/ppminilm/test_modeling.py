# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import unittest

import paddle

from paddlenlp.transformers import (
    PPMiniLMForMultipleChoice,
    PPMiniLMForQuestionAnswering,
    PPMiniLMForSequenceClassification,
    PPMiniLMModel,
)
from paddlenlp.transformers.ppminilm.configuration import PPMiniLMConfig

from ...testing_utils import slow
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    ids_tensor,
    random_attention_mask,
)


class PPMiniLMModelTester:
    def __init__(
        self,
        parent: PPMiniLMModelTest,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0,
        pool_act="tanh",
        num_labels=3,
        num_choices=4,
        scope=None,
        dropout=0.56,
    ):
        self.parent: PPMiniLMModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.pool_act = pool_act
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.dropout = dropout

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask

    def get_config(self) -> PPMiniLMConfig:
        return PPMiniLMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            pool_act=self.pool_act,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
        )

    def create_and_check_model(
        self,
        config: PPMiniLMConfig,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = PPMiniLMModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_for_multiple_choice(
        self,
        config: PPMiniLMConfig,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = PPMiniLMForMultipleChoice(config)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand([-1, self.num_choices, -1])
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
        )
        self.parent.assertEqual(result.shape, [self.batch_size, self.num_choices])

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = PPMiniLMForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
        )

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length])

    def create_and_check_for_sequence_classification(
        self,
        config: PPMiniLMConfig,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = PPMiniLMForSequenceClassification(config)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.shape, [self.batch_size, self.num_labels])

    def test_addition_params(self, config: PPMiniLMConfig, *args, **kwargs):
        config.num_labels = 7
        config.classifier_dropout = 0.98

        model = PPMiniLMForSequenceClassification(config)
        model.eval()

        self.parent.assertEqual(model.classifier.weight.shape, [config.hidden_size, 7])
        self.parent.assertEqual(model.dropout.p, 0.98)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


class PPMiniLMModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = PPMiniLMModel

    all_model_classes = (
        PPMiniLMModel,
        PPMiniLMForMultipleChoice,
        PPMiniLMForQuestionAnswering,
        PPMiniLMForSequenceClassification,
    )

    def setUp(self):
        super().setUp()

        self.model_tester = PPMiniLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PPMiniLMConfig, vocab_size=256, hidden_size=24)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_custom_params(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.test_addition_params(*config_and_inputs)

    def test_model_name_list(self):
        config = self.model_tester.get_config()
        model = self.base_model_class(config)
        self.assertTrue(len(model.model_name_list) != 0)

    @slow
    def test_params_compatibility_of_init_method(self):
        """test initing model with different params"""
        model: PPMiniLMForSequenceClassification = PPMiniLMForSequenceClassification.from_pretrained(
            "ppminilm-6l-768h", num_labels=4, dropout=0.3
        )
        assert model.num_labels == 4
        assert model.dropout.p == 0.3


class PPMiniLMModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = PPMiniLMModel

    @slow
    def test_inference_no_attention(self):
        model = PPMiniLMModel.from_pretrained("ppminilm-6l-768h")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)
        expected_slice = paddle.to_tensor(
            [
                [
                    [-0.79207015, 0.40036711, 1.18436682],
                    [-0.85833853, 0.34584877, 0.93867993],
                    [-0.97080499, 0.33460250, 0.69212830],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = PPMiniLMModel.from_pretrained("ppminilm-6l-768h")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)
        expected_slice = paddle.to_tensor(
            [
                [
                    [-0.79207015, 0.40036711, 1.18436682],
                    [-0.85833853, 0.34584877, 0.93867993],
                    [-0.97080499, 0.33460250, 0.69212830],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
