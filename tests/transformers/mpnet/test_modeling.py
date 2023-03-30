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

import unittest

import paddle

from paddlenlp.transformers import (
    MPNetConfig,
    MPNetForMaskedLM,
    MPNetForMultipleChoice,
    MPNetForQuestionAnswering,
    MPNetForSequenceClassification,
    MPNetForTokenClassification,
    MPNetModel,
)

from ...transformers.test_configuration_common import ConfigTester
from ...transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class MPNetModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
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
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return MPNetConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_mpnet_model(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MPNetModel(config=config)
        model.eval()
        result = model(input_ids, input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_mpnet_for_question_answering(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MPNetForQuestionAnswering(config=config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )
        start_logits, end_logits = result[0], result[1]
        self.parent.assertEqual(start_logits.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_logits.shape, [self.batch_size, self.seq_length])

    def create_and_check_mpnet_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = MPNetForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )
        if paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_labels])

    def create_and_check_mpnet_for_multiple_choice(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MPNetForMultipleChoice(config=config, num_choices=self.num_choices)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand((-1, self.num_choices, -1))
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand((-1, self.num_choices, -1))
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
        )
        if paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_choices])

    def create_and_check_mpnet_for_token_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = MPNetForTokenClassification(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        if paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_labels])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class MPNetModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = MPNetModel
    all_model_classes = (
        MPNetForMaskedLM,
        MPNetForMultipleChoice,
        MPNetForQuestionAnswering,
        MPNetForSequenceClassification,
        MPNetForTokenClassification,
        MPNetModel,
    )
    use_test_model_name_list = False
    return_dict = False
    use_labels = False

    def setUp(self):
        self.model_tester = MPNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MPNetConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mpnet_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mpnet_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mpnet_for_sequence_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mpnet_for_multiple_choice(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mpnet_for_token_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mpnet_for_question_answering(*config_and_inputs)
