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
    SkepConfig,
    SkepCrfForTokenClassification,
    SkepForSequenceClassification,
    SkepForTokenClassification,
    SkepModel,
    SkepPretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


class SkepModelTester:
    """Base Skep Model tester which can test:"""

    def __init__(
        self,
        parent,
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
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        num_classes=3,
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
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self) -> Tuple[SkepConfig, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_classes)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self) -> SkepConfig:
        return SkepConfig(
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
            num_class=self.num_classes,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
        )

    def create_and_check_model(
        self,
        config: SkepConfig,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = SkepModel(config)
        model.eval()

        result = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, return_dict=self.parent.return_dict
        )
        result = model(input_ids, token_type_ids=token_type_ids, return_dict=self.parent.return_dict)
        result = model(input_ids, return_dict=self.parent.return_dict)

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = SkepForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
            return_dict=self.parent.return_dict,
        )
        if not self.parent.return_dict and sequence_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))
        if sequence_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_classes])

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
        model = SkepForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            return_dict=self.parent.return_dict,
            labels=token_labels,
        )
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))
        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_classes])

    def create_and_check_for_crf_token_classification(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        input_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = SkepCrfForTokenClassification(config)
        model.eval()
        result = model(
            input_ids, token_type_ids=token_type_ids, return_dict=self.parent.return_dict, labels=token_labels
        )

        if paddle.is_tensor(result):
            result = [result]

        if token_labels is not None:
            self.parent.assertEqual(result[0].shape, [self.batch_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length])

    def create_and_check_model_cache(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = SkepModel(config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True, return_dict=self.parent.return_dict)
        past_key_values = outputs.past_key_values if self.parent.return_dict else outputs[2]

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), self.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], axis=-1)

        outputs = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            output_hidden_states=True,
            return_dict=self.parent.return_dict,
        )

        output_from_no_past = outputs[2][0]

        outputs = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=self.parent.return_dict,
        )

        output_from_past = outputs[2][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

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
class SkepModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = SkepModel
    return_dict = False
    use_labels = False
    use_test_inputs_embeds = True

    all_model_classes = (
        SkepModel,
        SkepCrfForTokenClassification,
        SkepForSequenceClassification,
        SkepForTokenClassification,
    )

    def setUp(self):
        self.model_tester = SkepModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_crf_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_crf_token_classification(*config_and_inputs)

    def test_for_model_cache(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_cache(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(SkepPretrainedModel.pretrained_init_configuration)[:1]:
            model = SkepModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class SkepModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_attention(self):
        model = SkepModel.from_pretrained("skep_ernie_1.0_large_ch")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 1024]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [0.31737554, 0.58842468, 0.43969756],
                    [0.20048048, 0.04142965, -0.2655520],
                    [0.49883127, -0.15263288, 0.46780178],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = SkepModel.from_pretrained("skep_ernie_1.0_large_ch")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 1024]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [0.31737554, 0.58842468, 0.43969756],
                    [0.20048048, 0.04142965, -0.2655520],
                    [0.49883127, -0.15263288, 0.46780178],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_past_key_value(self):
        model = SkepModel.from_pretrained("skep_ernie_1.0_large_ch")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)

        expected_shape = [1, 11, 1024]
        self.assertEqual(output[0].shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [0.31737363, 0.58842909, 0.43969074],
                    [0.20047806, 0.04142847, -0.26555336],
                    [0.49882850, -0.15263671, 0.46780348],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[0][:, 1:4, 1:4], expected_slice, atol=1e-4))

        # insert the past key value into model
        with paddle.no_grad():
            output = model(input_ids, use_cache=True, past_key_values=output.past_key_values, return_dict=True)
        expected_slice = paddle.to_tensor(
            [
                [
                    [0.29901379, 0.68195367, 0.62448436],
                    [0.18537062, 0.33085057, -0.04292759],
                    [0.38783669, -0.19946010, 0.24944240],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[0][:, 1:4, 1:4], expected_slice, atol=1e-4))
