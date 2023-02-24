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
from typing import Tuple

import paddle
from paddle import Tensor
from parameterized import parameterized_class

from paddlenlp.transformers import (
    NystromformerConfig,
    NystromformerForMaskedLM,
    NystromformerForMultipleChoice,
    NystromformerForQuestionAnswering,
    NystromformerForSequenceClassification,
    NystromformerForTokenClassification,
    NystromformerModel,
    NystromformerPretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


class NystromformerModelTester:
    """Base Nystromformer Model tester which can test:"""

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=8,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        type_sequence_label_size=2,
        conv_kernel_size=65,
        inv_coeff_init_option=False,
        layer_norm_eps=1e-05,
        num_landmarks=64,
        segment_means_seq_len=64,
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
        self.type_sequence_label_size = type_sequence_label_size
        self.conv_kernel_size = conv_kernel_size
        self.inv_coeff_init_option = inv_coeff_init_option
        self.layer_norm_eps = layer_norm_eps
        self.num_landmarks = num_landmarks
        self.segment_means_seq_len = segment_means_seq_len
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def get_config(self) -> NystromformerConfig:
        return NystromformerConfig(
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
            segment_means_seq_len=self.segment_means_seq_len,
            num_landmarks=self.num_landmarks,
            conv_kernel_size=self.conv_kernel_size,
            inv_coeff_init_option=self.inv_coeff_init_option,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            num_class=self.num_labels,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
        )

    def prepare_config_and_inputs(self) -> Tuple[NystromformerConfig, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_model(
        self,
        config: NystromformerConfig,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        # 1. test model instantiation and forward w/o token_type_ids
        model = NystromformerModel(config)
        model.eval()

        result = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, return_dict=self.parent.return_dict
        )
        result = model(input_ids, token_type_ids=token_type_ids, return_dict=self.parent.return_dict)
        result = model(input_ids, return_dict=self.parent.return_dict)

        # nystromformer only return one tensor: last_hidden_state
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

        # 2. test forward with chunk computing
        config.chunk_size_feed_forward = True
        model_with_chunk = NystromformerModel(config)
        model_with_chunk.load_dict(model.state_dict())
        model_with_chunk.eval()
        result_with_chunk = model_with_chunk(input_ids, return_dict=self.parent.return_dict)
        self.parent.assertTrue(paddle.allclose(result[0], result_with_chunk[0], atol=1e-4))
        model.config.chunk_size_feed_forward = False

        # 3. test nystrom attention
        config.segment_means_seq_len = input_ids.shape[1]
        config.num_landmarks = 2
        model_with_nystrom = NystromformerModel(config)
        model_with_nystrom.eval()
        result_with_nystrom = model_with_nystrom(input_ids, return_dict=self.parent.return_dict)
        self.parent.assertEqual(result_with_nystrom[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_sequence_classification(
        self,
        config: NystromformerConfig,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        config.num_labels = self.type_sequence_label_size
        model = NystromformerForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
            return_dict=self.parent.return_dict,
        )
        if not self.parent.return_dict and sequence_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result[0]))

        if sequence_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.type_sequence_label_size])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        config.num_labels = self.num_labels
        model = NystromformerForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            return_dict=self.parent.return_dict,
            labels=token_labels,
        )
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result[0]))
        if token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_labels])

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        config.num_labels = self.vocab_size
        model = NystromformerForMaskedLM(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            return_dict=self.parent.return_dict,
            labels=token_labels,
        )
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result[0]))
        if token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        config.num_labels = self.num_choices
        model = NystromformerForMultipleChoice(config)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand([-1, self.num_choices, -1])
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
            return_dict=self.parent.return_dict,
        )
        if choice_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_choices])

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        config.num_labels = self.num_labels
        model = NystromformerForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            return_dict=self.parent.return_dict,
        )

        if sequence_labels is not None:
            start_logits, end_logits = result[1], result[2]
        else:
            start_logits, end_logits = result[0], result[1]

        self.parent.assertEqual(start_logits.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_logits.shape, [self.batch_size, self.seq_length])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@parameterized_class(
    ("return_dict", "use_labels"),
    [
        [False, False],
        [False, True],
        [True, False],
        [True, True],
    ],
)
class NystromformerModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = NystromformerModel
    return_dict = False
    use_labels = False

    all_model_classes = (
        NystromformerModel,
        NystromformerForSequenceClassification,
        NystromformerForTokenClassification,
    )

    def setUp(self):
        self.model_tester = NystromformerModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(NystromformerPretrainedModel.pretrained_init_configuration)[:1]:
            model = NystromformerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class NystromformerModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_attention(self):
        model = NystromformerModel.from_pretrained("nystromformer-base-zh")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [input_ids.shape[0], input_ids.shape[1], model.config.hidden_size]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [0.27683097, -2.19216943, -0.23561366],
                    [0.10705502, -2.06556797, -0.07792263],
                    [0.53340679, -2.20003223, -0.07504901],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = NystromformerModel.from_pretrained("nystromformer-base-zh")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [input_ids.shape[0], input_ids.shape[1], model.config.hidden_size]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [-0.46736166, -1.27038229, 0.81337416],
                    [-0.59629452, -1.13692689, 0.81597191],
                    [-0.55872959, -1.07646871, 0.72584474],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
