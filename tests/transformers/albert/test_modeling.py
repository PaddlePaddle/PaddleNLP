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
from parameterized import parameterized_class
import paddle
from paddle import Tensor

from paddlenlp.transformers import (
    AlbertPretrainedModel,
    AlbertForMaskedLM,
    AlbertForMultipleChoice,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    AlbertForTokenClassification,
    AlbertModel,
)
from ..test_modeling_common import ids_tensor, random_attention_mask, ModelTesterMixin
from ...testing_utils import slow


class AlbertModelTester:

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
        self.vocab_size = 99
        self.embedding_size = 16
        self.hidden_size = 36
        self.num_hidden_layers = 6
        self.num_hidden_groups = 6
        self.num_attention_heads = 6
        self.intermediate_size = 37
        self.inner_group_num = 1
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.initializer_range = 0.02
        self.layer_norm_eps = 1e-12
        self.pad_token_id = 0,
        self.bos_token_id = 2,
        self.eos_token_id = 3,
        self.add_pooling_layer = True
        self.type_sequence_label_size = 2
        self.num_classes = 3
        self.num_choices = 4
        self.scope = None

    def prepare_config_and_inputs(self):
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
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
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
            "num_hidden_groups": self.num_hidden_groups,
        }

    def create_and_check_model(self, config, input_ids: Tensor,
                               token_type_ids: Tensor, input_mask: Tensor,
                               sequence_labels: Tensor, token_labels: Tensor,
                               choice_labels: Tensor):
        model = AlbertModel(**config)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       return_dict=self.parent.return_dict)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids, return_dict=self.parent.return_dict)
        self.parent.assertEqual(
            result[0].shape,
            [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape,
                                [self.batch_size, self.hidden_size])

    def create_and_check_for_masked_lm(self, config, input_ids: Tensor,
                                       token_type_ids: Tensor,
                                       input_mask: Tensor,
                                       sequence_labels: Tensor,
                                       token_labels: Tensor,
                                       choice_labels: Tensor):
        model = AlbertForMaskedLM(AlbertModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       labels=token_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]
        self.parent.assertEqual(
            result[0].shape,
            [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_question_answering(self, config, input_ids: Tensor,
                                                token_type_ids: Tensor,
                                                input_mask: Tensor,
                                                sequence_labels: Tensor,
                                                token_labels: Tensor,
                                                choice_labels: Tensor):
        model = AlbertForQuestionAnswering(AlbertModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       start_positions=sequence_labels,
                       end_positions=sequence_labels,
                       return_dict=self.parent.return_dict)

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape,
                                [self.batch_size, self.seq_length])
        self.parent.assertEqual(result[1].shape,
                                [self.batch_size, self.seq_length])

    def create_and_check_for_sequence_classification(
            self, config, input_ids: Tensor, token_type_ids: Tensor,
            input_mask: Tensor, sequence_labels: Tensor, token_labels: Tensor,
            choice_labels: Tensor):
        model = AlbertForSequenceClassification(AlbertModel(**config),
                                                num_classes=self.num_classes)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       labels=sequence_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]
        self.parent.assertEqual(result[0].shape,
                                [self.batch_size, self.num_classes])

    def create_and_check_for_token_classification(
            self, config, input_ids: Tensor, token_type_ids: Tensor,
            input_mask: Tensor, sequence_labels: Tensor, token_labels: Tensor,
            choice_labels: Tensor):
        model = AlbertForTokenClassification(AlbertModel(**config),
                                             num_classes=self.num_classes)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids,
                       labels=token_labels,
                       return_dict=self.parent.return_dict)

        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(
            result[0].shape,
            [self.batch_size, self.seq_length, self.num_classes])

    def create_and_check_for_multiple_choice(self, config, input_ids: Tensor,
                                             token_type_ids: Tensor,
                                             input_mask: Tensor,
                                             sequence_labels: Tensor,
                                             token_labels: Tensor,
                                             choice_labels: Tensor):
        model = AlbertForMultipleChoice(AlbertModel(**config))
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
                       labels=choice_labels,
                       return_dict=self.parent.return_dict)
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]
        self.parent.assertEqual(result[0].shape,
                                [self.batch_size, self.num_choices])

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
class AlbertModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = AlbertModel
    use_labels = False
    return_dict = False

    all_model_classes = (
        AlbertModel,
        AlbertForMaskedLM,
        AlbertForMultipleChoice,
        AlbertForSequenceClassification,
        AlbertForTokenClassification,
        AlbertForQuestionAnswering,
    )

    def setUp(self):
        self.model_tester = AlbertModelTester(self)

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

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(
                AlbertPretrainedModel.pretrained_init_configuration.keys())[:1]:
            model = AlbertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class AlbertModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = AlbertModel.from_pretrained("albert-base-v2")
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)
        expected_slice = paddle.to_tensor([[[-0.6513, 1.5035, -0.2766],
                                            [-0.6515, 1.5046, -0.2780],
                                            [-0.6512, 1.5049, -0.2784]]])

        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))