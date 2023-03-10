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

import inspect
import unittest

from paddlenlp.transformers import (
    RoFormerv2Config,
    RoFormerv2ForMaskedLM,
    RoFormerv2ForMultipleChoice,
    RoFormerv2ForQuestionAnswering,
    RoFormerv2ForSequenceClassification,
    RoFormerv2ForTokenClassification,
    RoFormerv2Model,
    RoFormerv2PretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


class RoFormerv2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        initializer_range=0.02,
        type_sequence_label_size=2,
        num_labels=3,
        num_classes=3,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=8,
        hidden_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        act_dropout=0,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,
        rotary_value=False,
        use_bias=False,
        epsilon=1e-12,
        normalize_before=False,
        num_choices=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.initializer_range = initializer_range
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.act_dropout = act_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.pad_token_id = pad_token_id
        self.rotary_value = rotary_value
        self.use_bias = use_bias
        self.epsilon = epsilon
        self.normalize_before = normalize_before
        self.num_choices = num_choices

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

    def get_config(self):
        return RoFormerv2Config(
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
            pad_token_id=self.pad_token_id,
            rotary_value=self.rotary_value,
            use_bias=self.use_bias,
            epsilon=self.epsilon,
            normalize_before=self.normalize_before,
            num_choices=self.num_choices,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):

        model = RoFormerv2Model(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RoFormerv2ForMaskedLM(config)
        model.eval()

        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
        )
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RoFormerv2ForMultipleChoice(config)
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
        model = RoFormerv2ForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )

        start_logits = result[0]
        end_logits = result[1]

        self.parent.assertEqual(start_logits.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_logits.shape, [self.batch_size, self.seq_length])

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RoFormerv2ForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )

        self.parent.assertEqual(result.shape, [self.batch_size, self.num_choices])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = RoFormerv2ForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )

        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.num_choices])

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


class RoFormerv2ModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = RoFormerv2Model
    return_dict: bool = False
    use_labels: bool = False

    all_model_classes = (
        RoFormerv2Model,
        RoFormerv2ForMaskedLM,
        RoFormerv2ForSequenceClassification,
        RoFormerv2ForTokenClassification,
        RoFormerv2ForQuestionAnswering,
        RoFormerv2ForMultipleChoice,
    )

    def setUp(self):
        self.model_tester = RoFormerv2ModelTester(self)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["input_ids"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

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
        for model_name in list(RoFormerv2PretrainedModel.pretrained_init_configuration)[:1]:
            model = RoFormerv2Model.from_pretrained(model_name)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
