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
from paddle import Tensor
from parameterized import parameterized_class

from paddlenlp.transformers import (
    UIEM,
    ErnieMConfig,
    ErnieMForMultipleChoice,
    ErnieMForQuestionAnswering,
    ErnieMForSequenceClassification,
    ErnieMForTokenClassification,
    ErnieMModel,
    ErnieMPretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


class ErnieMModelTester:
    """Base ErnieM Model tester which can test:"""

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_position_ids=True,
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
        self.use_position_ids = use_position_ids
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

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        attention_mask = None
        if self.use_input_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        position_ids = None
        if self.use_position_ids:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones

        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_classes)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, position_ids, attention_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return ErnieMConfig(
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

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, position_ids, attention_mask, _, _, _ = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config: ErnieMConfig,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = ErnieMModel(config)
        model.eval()

        result = model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=self.parent.return_dict
        )
        result = model(input_ids, position_ids=position_ids, return_dict=self.parent.return_dict)
        result = model(input_ids, attention_mask=attention_mask, return_dict=self.parent.return_dict)

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_for_sequence_classification(
        self,
        config: ErnieMConfig,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = ErnieMForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=sequence_labels,
            return_dict=self.parent.return_dict,
        )

        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_classes])

    def create_and_check_for_question_answering(
        self,
        config: ErnieMConfig,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = ErnieMForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            return_dict=self.parent.return_dict,
        )

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_classes])

    def create_and_check_for_uie(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = UIEM(config)
        model.eval()
        start_prob, end_prob = model(
            input_ids,
            attention_mask=input_mask,
        )

        self.parent.assertEqual(start_prob.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_prob.shape, [self.batch_size, self.seq_length])

    def create_and_check_for_token_classification(
        self,
        config: ErnieMConfig,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = ErnieMForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=token_labels,
            return_dict=self.parent.return_dict,
        )
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_classes])

    def create_and_check_for_multiple_choice(
        self,
        config: ErnieMConfig,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = ErnieMForMultipleChoice(config)
        model.eval()

        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_position_ids = position_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_attention_mask = attention_mask.unsqueeze(1).expand([-1, self.num_choices, -1])

        result = model(
            multiple_choice_inputs_ids,
            position_ids=multiple_choice_position_ids,
            attention_mask=multiple_choice_attention_mask,
            labels=choice_labels,
            return_dict=self.parent.return_dict,
        )
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))
        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_choices])

    def create_and_check_model_cache(
        self, config: ErnieMConfig, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = ErnieMModel(config)
        model.eval()

        input_ids = ids_tensor((self.batch_size, self.seq_length), self.vocab_size)

        # create tensors for past_key_values of shape [batch_size, num_heads, seq_length, head_size]
        embed_size_per_head = self.hidden_size // self.num_attention_heads
        key_tensor = floats_tensor((self.batch_size, self.num_attention_heads, self.seq_length, embed_size_per_head))
        values_tensor = floats_tensor(
            (self.batch_size, self.num_attention_heads, self.seq_length, embed_size_per_head)
        )
        past_key_values = (
            (
                key_tensor,
                values_tensor,
            ),
        ) * self.num_hidden_layers

        # create fully-visible attention mask for input_ids only and input_ids + past
        attention_mask = paddle.ones([self.batch_size, self.seq_length])
        attention_mask_with_past = paddle.ones([self.batch_size, self.seq_length * 2])

        outputs_with_cache = model(
            input_ids,
            attention_mask=attention_mask_with_past,
            past_key_values=past_key_values,
            return_dict=self.parent.return_dict,
        )
        outputs_without_cache = model(input_ids, attention_mask=attention_mask, return_dict=self.parent.return_dict)

        # last_hidden_state should have the same shape but different values when given past_key_values
        if self.parent.return_dict:
            self.parent.assertEqual(
                outputs_with_cache.last_hidden_state.shape, outputs_without_cache.last_hidden_state.shape
            )
            self.parent.assertFalse(
                paddle.allclose(outputs_with_cache.last_hidden_state, outputs_without_cache.last_hidden_state)
            )
        else:
            outputs_with_cache, _ = outputs_with_cache
            outputs_without_cache, _ = outputs_without_cache
            self.parent.assertEqual(outputs_with_cache.shape, outputs_without_cache.shape)
            self.parent.assertFalse(paddle.allclose(outputs_with_cache, outputs_without_cache))


@parameterized_class(
    ("return_dict", "use_labels"),
    [
        [False, False],
        [False, True],
        [True, False],
        [True, True],
    ],
)
class ErnieMModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = ErnieMModel
    use_labels = False
    return_dict = False
    use_inputs_embeds = True

    all_model_classes = (
        ErnieMModel,
        ErnieMForSequenceClassification,
        ErnieMForTokenClassification,
        ErnieMForQuestionAnswering,
        ErnieMForMultipleChoice,
        UIEM,
    )

    def setUp(self):
        self.model_tester = ErnieMModelTester(self)

        # set attribute in setUp to overwrite the static attribute
        self.test_resize_embeddings = False

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_uie(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_uie(*config_and_inputs)

    def test_for_multi_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_model_cache(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_cache(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(ErnieMPretrainedModel.pretrained_init_configuration)[:1]:
            model = ErnieMModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class ErnieMModelIntegrationTest(unittest.TestCase):
    base_model_class = ErnieMPretrainedModel
    hf_remote_test_model_path = "PaddleCI/tiny-random-ernie-m"

    @slow
    def test_inference_no_attention(self):
        model = ErnieMModel.from_pretrained("ernie-m-base")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [-0.02920425, -0.00768885, -0.10219190],
                    [-0.10798159, 0.02311476, -0.17285497],
                    [0.05675533, 0.01330730, -0.06826267],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = ErnieMModel.from_pretrained("ernie-m-base")
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
                    [-0.02920425, -0.00768885, -0.10219190],
                    [-0.10798159, 0.02311476, -0.17285497],
                    [0.05675533, 0.01330730, -0.06826267],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_past_key_value(self):
        model = ErnieMModel.from_pretrained("ernie-m-base")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)

        expected_shape = [1, 11, 768]
        self.assertEqual(output[0].shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [-0.02920425, -0.00768885, -0.10219190],
                    [-0.10798159, 0.02311476, -0.17285497],
                    [0.05675533, 0.01330730, -0.06826267],
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
                    [0.05163988, -0.07475190, 0.06332156],
                    [0.03051429, -0.01377687, -0.12024689],
                    [0.03379946, 0.00674286, 0.08079184],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[0][:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
