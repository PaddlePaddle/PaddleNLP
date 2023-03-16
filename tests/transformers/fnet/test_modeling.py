# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PaddleNLP FNet model. """


import unittest

from paddlenlp.transformers import (
    FNetConfig,
    FNetForMaskedLM,
    FNetForMultipleChoice,
    FNetForNextSentencePrediction,
    FNetForPreTraining,
    FNetForQuestionAnswering,
    FNetForSequenceClassification,
    FNetForTokenClassification,
    FNetModel,
    FNetPretrainedModel,
)
from paddlenlp.transformers.configuration_utils import PretrainedConfig

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor


class FNetModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
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
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels

    def get_config(self) -> FNetConfig:
        return FNetConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            tpu_short_seq_length=self.seq_length,
        )

    def create_and_check_model(self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels):
        model = FNetModel(config=config)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids, return_dict=True)
        self.parent.assertEqual(
            result["last_hidden_state"].shape, [self.batch_size, self.seq_length, self.hidden_size]
        )

    def create_and_check_for_pretraining(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        model = FNetForPreTraining(config=config)
        model.eval()
        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            labels=token_labels,
            next_sentence_label=sequence_labels,
            return_dict=True,
        )
        self.parent.assertEqual(result["prediction_logits"].shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertEqual(result["seq_relationship_logits"].shape, [self.batch_size, 2])

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        model = FNetForMaskedLM(config=config)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, labels=token_labels, return_dict=True)
        self.parent.assertEqual(result["prediction_logits"].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_next_sentence_prediction(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        model = FNetForNextSentencePrediction(config=config)
        model.eval()
        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            next_sentence_label=sequence_labels,
            return_dict=True,
        )
        self.parent.assertEqual(result["logits"].shape, [self.batch_size, 2])

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        model = FNetForQuestionAnswering(config=config)
        model.eval()
        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            return_dict=True,
        )
        self.parent.assertEqual(result["start_logits"].shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(result["end_logits"].shape, [self.batch_size, self.seq_length])

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = FNetForSequenceClassification(config)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, labels=sequence_labels, return_dict=True)
        self.parent.assertEqual(result["logits"].shape, [self.batch_size, self.num_labels])

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = FNetForTokenClassification(config=config)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, labels=token_labels, return_dict=True)
        self.parent.assertEqual(result["logits"].shape, [self.batch_size, self.seq_length, self.num_labels])

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = FNetForMultipleChoice(config=config)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        result = model(
            multiple_choice_inputs_ids,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
            return_dict=True,
        )
        self.parent.assertEqual(result["logits"].shape, [self.batch_size, self.num_choices])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids}
        return config, inputs_dict

    def create_and_check_config_and_inputs_for_common(self, config):

        self.parent.assertTrue(isinstance(config, FNetConfig))
        self.parent.assertTrue(issubclass(FNetConfig, PretrainedConfig))
        self.parent.assertTrue(isinstance(config, PretrainedConfig))


class FNetModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        FNetModel,
        FNetForPreTraining,
        FNetForMaskedLM,
        FNetForNextSentencePrediction,
        FNetForMultipleChoice,
        FNetForQuestionAnswering,
        FNetForSequenceClassification,
        FNetForTokenClassification,
    )

    base_model_class = FNetModel
    return_dict = False
    use_labels = False
    # Skip Tests
    test_pruning = False
    test_head_masking = False
    test_pruning = False

    # Overriden Tests
    def test_attention_outputs(self):
        pass

    def setUp(self):
        self.model_tester = FNetModelTester(self)

    def test_config_for_common(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_config_and_inputs_for_common(config)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(FNetPretrainedModel.pretrained_init_configuration)[:1]:
            model = FNetModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
