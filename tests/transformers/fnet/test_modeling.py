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
)

from ..test_modeling_common import ModelTesterMixin, ids_tensor


class FnetModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=False,
        use_token_type_ids=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=4,
        intermediate_size=64,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        add_pooling_layer=True,
        num_labels=2,
        num_classes=3,
        return_dict=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.vocab_size = vocab_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.add_pooling_layer = add_pooling_layer
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        num_labels = self.num_labels
        num_classes = self.num_classes
        config = self.get_config()
        return_dict = self.return_dict
        return (config, input_ids, token_type_ids, num_classes, num_labels, return_dict)

    def get_config(self):
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
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            add_pooling_layer=self.add_pooling_layer,
            # num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            num_classes,
            num_labels,
            return_dict,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        num_classes,
        num_labels,
        return_dict,
    ):
        model = FNetModel(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        self.parent.assertEqual(
            result["last_hidden_state"].shape, [self.batch_size, self.seq_length, self.hidden_size]
        )

    def create_and_check_sequence_classification_model(
        self,
        config,
        input_ids,
        token_type_ids,
        num_classes,
        num_labels,
        return_dict,
    ):
        model = FNetForSequenceClassification(config, num_classes)
        model.eval()
        result = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        self.parent.assertEqual(result["logits"].shape, [self.batch_size, self.num_classes])

    def create_and_check_token_classification_model(
        self,
        config,
        input_ids,
        token_type_ids,
        num_classes,
        num_labels,
        return_dict,
    ):
        model = FNetForTokenClassification(config, num_classes)
        model.eval()
        result = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        self.parent.assertEqual(result["logits"].shape, [self.batch_size, self.seq_length, self.num_classes])

    def create_and_check_masked_lm_model(
        self,
        config,
        input_ids,
        token_type_ids,
        num_classes,
        num_labels,
        return_dict,
    ):
        model = FNetForMaskedLM(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        self.parent.assertEqual(result["prediction_logits"].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_pretraining_model(
        self,
        config,
        input_ids,
        token_type_ids,
        num_classes,
        num_labels,
        return_dict,
    ):
        model = FNetForPreTraining(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        self.parent.assertEqual(result["prediction_logits"].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_next_sentence_prediction_model(
        self,
        config,
        input_ids,
        token_type_ids,
        num_classes,
        num_labels,
        return_dict,
    ):
        model = FNetForNextSentencePrediction(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        self.parent.assertEqual(result["seq_relationship_logits"].shape, [self.batch_size, 2])

    def create_and_check_multiple_chocie_model(
        self,
        config,
        input_ids,
        token_type_ids,
        num_classes,
        num_labels,
        return_dict,
    ):
        model = FNetForMultipleChoice(config)
        model.eval()
        input_ids = ids_tensor([self.batch_size, self.num_classes, self.seq_length], self.vocab_size)
        token_type_ids = ids_tensor([self.batch_size, self.num_classes, self.seq_length], self.type_vocab_size)
        result = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        self.parent.assertEqual(result["logits"].shape, [self.batch_size, self.num_classes])

    def create_and_check_question_answering_model(
        self,
        config,
        input_ids,
        token_type_ids,
        num_classes,
        num_labels,
        return_dict,
    ):
        model = FNetForQuestionAnswering(config, num_labels)
        model.eval()
        result = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        self.parent.assertEqual(result["start_logits"].shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(result["end_logits"].shape, [self.batch_size, self.seq_length])


class FnetModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = FNetModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = False

    all_model_classes = (FNetModel,)

    def setUp(self):
        self.model_tester = FnetModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_sequence_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_sequence_classification_model(*config_and_inputs)

    def test_pretraining_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_pretraining_model(*config_and_inputs)

    def test_masked_lm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_masked_lm_model(*config_and_inputs)

    def test_next_sentence_prediction_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_next_sentence_prediction_model(*config_and_inputs)

    def test_multiple_chocie_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_multiple_chocie_model(*config_and_inputs)

    def test_token_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_token_classification_model(*config_and_inputs)

    def test_question_answering_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_question_answering_model(*config_and_inputs)
