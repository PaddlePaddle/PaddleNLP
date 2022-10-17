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

from dataclasses import dataclass
import unittest
from parameterized import parameterized_class

import paddle

from paddlenlp.transformers import (
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForPretraining,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    ElectraModel,
    ElectraPretrainedModel,
)
from ..test_modeling_common import ids_tensor, floats_tensor, random_attention_mask, ModelTesterMixin
from ...testing_utils import slow


class ElectraModelTester:

    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = True
        self.use_inputs_embeds = False
        self.vocab_size = 99
        self.embedding_size = 32
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.initializer_range = 0.02
        self.pad_token_id = 0
        self.layer_norm_eps = 1e-12
        self.type_sequence_label_size = 2
        self.num_classes = 3
        self.num_choices = 2

    def prepare_config_and_inputs(self):
        input_ids = None
        inputs_embeds = None
        if self.use_inputs_embeds:
            inputs_embeds = floats_tensor(
                [self.batch_size, self.seq_length, self.embedding_size])
        else:
            input_ids = ids_tensor([self.batch_size, self.seq_length],
                                   self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask(
                [self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length],
                                        self.type_vocab_size)

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
        return config, input_ids, token_type_ids, input_mask, inputs_embeds, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
        }

    def create_and_check_electra_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        inputs_embeds,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ElectraModel(**config)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       inputs_embeds=inputs_embeds,
                       return_dict=self.parent.return_dict)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids, return_dict=self.parent.return_dict)

        if paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(
            result[0].shape,
            [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_electra_model_cache(self, config, input_ids,
                                             token_type_ids, input_mask,
                                             inputs_embeds, sequence_labels,
                                             token_labels, choice_labels):
        model = ElectraModel(**config)
        model.eval()

        input_ids = ids_tensor((self.batch_size, self.seq_length),
                               self.vocab_size)
        input_token_types = ids_tensor([self.batch_size, self.seq_length],
                                       self.type_vocab_size)

        # first forward pass
        first_pass_outputs = model(input_ids,
                                   token_type_ids=input_token_types,
                                   use_cache=True,
                                   return_dict=True)
        past_key_values = first_pass_outputs.past_key_values

        # fully-visible attention mask
        attention_mask = paddle.ones([self.batch_size, self.seq_length * 2])

        # second forward pass with past_key_values with visible mask
        second_pass_outputs = model(input_ids,
                                    token_type_ids=input_token_types,
                                    attention_mask=attention_mask,
                                    past_key_values=past_key_values,
                                    return_dict=self.parent.return_dict)

        # last_hidden_state should have the same shape but different values when given past_key_values
        if self.parent.return_dict:
            self.parent.assertEqual(second_pass_outputs.last_hidden_state.shape,
                                    first_pass_outputs.last_hidden_state.shape)
            self.parent.assertFalse(
                paddle.allclose(second_pass_outputs.last_hidden_state,
                                first_pass_outputs.last_hidden_state))
        else:
            self.parent.assertEqual(second_pass_outputs.shape,
                                    first_pass_outputs[0].shape)
            self.parent.assertFalse(
                paddle.allclose(second_pass_outputs, first_pass_outputs[0]))

    def create_and_check_electra_for_masked_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        inputs_embeds,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ElectraForMaskedLM(ElectraModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       inputs_embeds=inputs_embeds,
                       labels=token_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(
            result[0].shape,
            [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_electra_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        inputs_embeds,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ElectraForTokenClassification(ElectraModel(**config),
                                              num_classes=self.num_classes)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       inputs_embeds=inputs_embeds,
                       labels=token_labels,
                       return_dict=self.parent.return_dict)

        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(
            result[0].shape,
            [self.batch_size, self.seq_length, self.num_classes])

    def create_and_check_electra_for_pretraining(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        inputs_embeds,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ElectraForPretraining(ElectraModel(**config))
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        self.parent.assertEqual(result.logits.shape,
                                (self.batch_size, self.seq_length))

    def create_and_check_electra_for_sequence_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        inputs_embeds,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ElectraForSequenceClassification(
            ElectraModel(**config), num_classes=self.type_sequence_label_size)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       inputs_embeds=inputs_embeds,
                       labels=sequence_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(
            result[0].shape, [self.batch_size, self.type_sequence_label_size])

    def create_and_check_electra_for_question_answering(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        inputs_embeds,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ElectraForQuestionAnswering(ElectraModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       inputs_embeds=inputs_embeds,
                       start_positions=sequence_labels,
                       end_positions=sequence_labels,
                       return_dict=self.parent.return_dict)

        if token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape,
                                [self.batch_size, self.seq_length])
        self.parent.assertEqual(result[1].shape,
                                [self.batch_size, self.seq_length])

    def create_and_check_electra_for_multiple_choice(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        inputs_embeds,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ElectraForMultipleChoice(ElectraModel(**config),
                                         num_choices=self.num_choices)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(
            [-1, self.num_choices, -1])
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(
            [-1, self.num_choices, -1])
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(
            [-1, self.num_choices, -1])
        result = model(multiple_choice_inputs_ids,
                       attention_mask=multiple_choice_input_mask,
                       token_type_ids=multiple_choice_token_type_ids,
                       inputs_embeds=inputs_embeds,
                       labels=choice_labels,
                       return_dict=self.parent.return_dict)

        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape,
                                [self.batch_size, self.num_choices])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            inputs_embeds,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
            "inputs_embeds": inputs_embeds
        }
        return config, inputs_dict


@parameterized_class(("return_dict", "use_labels", "use_inputs_embeds"), [
    [False, False, True],
    [False, False, False],
    [False, True, False],
    [True, False, False],
    [True, True, False],
])
class ElectraModelTest(ModelTesterMixin, unittest.TestCase):
    test_resize_embeddings = False
    base_model_class = ElectraModel

    use_labels = False
    return_dict = False

    all_model_classes = (
        ElectraModel,
        ElectraForMaskedLM,
        ElectraForMultipleChoice,
        ElectraForTokenClassification,
        ElectraForSequenceClassification,
        ElectraForQuestionAnswering,
    )

    def setUp(self):
        self.model_tester = ElectraModelTester(self)

        # set attribute in setUp to overwrite the static attribute
        self.test_resize_embeddings = False

    def test_electra_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_electra_model(*config_and_inputs)

    def test_electra_model_cache(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_electra_model_cache(
            *config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_electra_for_masked_lm(
            *config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_electra_for_token_classification(
            *config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_electra_for_sequence_classification(
            *config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_electra_for_question_answering(
            *config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_electra_for_multiple_choice(
            *config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(
                ElectraPretrainedModel.pretrained_init_configuration)[:1]:
            model = ElectraModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class ElectraModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = ElectraModel.from_pretrained("electra-small")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        output = model(input_ids, attention_mask=attention_mask)
        expected_shape = [1, 11, 256]
        self.assertEqual(output.shape, expected_shape)
        expected_slice = paddle.to_tensor([[[0.4471, 0.6821, -0.3265],
                                            [0.4627, 0.5255, -0.3668],
                                            [0.4532, 0.3313, -0.4344]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))
