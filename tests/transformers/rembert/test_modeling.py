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

import paddle

from paddlenlp.transformers import (
    RemBertConfig,
    RemBertForMaskedLM,
    RemBertForMultipleChoice,
    RemBertForQuestionAnswering,
    RemBertForSequenceClassification,
    RemBertForTokenClassification,
    RemBertModel,
    RemBertPretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor


class RemBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        vocab_size=99,
        input_embedding_size=64,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        layer_norm_eps=1e-12,
        num_classes=2,
        num_choices=1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.vocab_size = vocab_size
        self.input_embedding_size = input_embedding_size
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
        self.layer_norm_eps = layer_norm_eps
        self.num_classes = num_classes
        self.num_choices = num_choices

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = paddle.ones([self.batch_size, self.seq_length], dtype="int32")

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask

    def get_config(self):
        return RemBertConfig(
            vocab_size=self.vocab_size,
            input_embedding_size=self.input_embedding_size,
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
            layer_norm_eps=self.layer_norm_eps,
            num_classes=self.num_classes,
            num_choices=self.num_choices,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, token_type_ids) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RemBertModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_masked_lm_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RemBertForMaskedLM(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_question_answering_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RemBertForQuestionAnswering(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, 1])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, 1])

    def create_and_check_sequence_classification_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RemBertForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.shape, [self.batch_size, self.num_classes])

    def create_and_check_multiple_choice_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RemBertForMultipleChoice(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.shape, [self.batch_size, self.num_choices])

    def create_and_check_token_classification_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RemBertForTokenClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.num_classes])


class RemBertModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = RemBertModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = False

    all_model_classes = (
        RemBertModel,
        RemBertForMaskedLM,
        RemBertForQuestionAnswering,
        RemBertForSequenceClassification,
        RemBertForMultipleChoice,
        RemBertForTokenClassification,
    )

    def setUp(self):
        self.model_tester = RemBertModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_masked_lm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_masked_lm_model(*config_and_inputs)

    def test_question_answering_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_question_answering_model(*config_and_inputs)

    def test_sequence_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_sequence_classification_model(*config_and_inputs)

    def test_multiple_choice_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_multiple_choice_model(*config_and_inputs)

    def test_token_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_token_classification_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(RemBertPretrainedModel.pretrained_init_configuration)[:1]:
            model = RemBertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
