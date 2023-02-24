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
from parameterized import parameterized_class

from paddlenlp.transformers import (
    MobileBertConfig,
    MobileBertForQuestionAnswering,
    MobileBertForSequenceClassification,
    MobileBertModel,
    PretrainedModel,
)

from ...testing_utils import slow
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    ids_tensor,
    random_attention_mask,
)


class MobileBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        embedding_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
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
        self.embedding_size = embedding_size
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
        self.scope = scope

    def prepare_config_and_inputs(self):
        inputs = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, inputs, token_type_ids, input_mask, sequence_labels

    def get_config(self):
        return MobileBertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels):
        model = MobileBertModel(config=config)
        model.eval()
        result = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, return_dict=self.parent.return_dict
        )
        result = model(input_ids, return_dict=self.parent.return_dict)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_for_question_answering(self, config, input_ids, token_type_ids, input_mask, sequence_labels):
        model = MobileBertForQuestionAnswering(config=config)
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

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels
    ):
        config.num_labels = self.num_labels
        model = MobileBertForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
            return_dict=self.parent.return_dict,
        )
        if sequence_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_labels])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
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
class MobileBertModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = MobileBertModel
    return_dict = False
    use_labels = False
    is_decoder = True

    all_model_classes = (
        MobileBertModel,
        MobileBertForSequenceClassification,
        MobileBertForQuestionAnswering,
    )

    def setUp(self):
        self.model_tester = MobileBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MobileBertConfig, vocab_size=256, hidden_size=24)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_model_from_pretrained(self):
        for model_name in MobileBertModel.pretrained_init_configuration.keys():
            model = MobileBertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class MobileBertModelIntegrationTest(unittest.TestCase, ModelTesterPretrainedMixin):
    base_model_class: PretrainedModel = MobileBertModel
    hf_remote_test_model_path: str = "google/mobilebert-uncased"
    paddlehub_remote_test_model_name: str = "mobilebert-uncased"

    @slow
    def test_inference_no_attention(self):
        model = MobileBertModel.from_pretrained("mobilebert-uncased")
        model.eval()
        input_ids = paddle.to_tensor([[101, 7110, 1005, 1056, 2023, 11333, 17413, 1029, 102]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 9, 512]
        self.assertEqual(output.shape, expected_shape)
        expected_slice = paddle.to_tensor(
            [
                [
                    [-2.4736526e07, 8.2691656e04, 1.6521838e05],
                    [-5.7541704e-01, 3.9056022e00, 4.4011507e00],
                    [2.6047359e00, 1.5677652e00, -1.7324188e-01],
                ]
            ]
        )
        lower_bound = paddle.all((expected_slice / output[..., :3, :3]) >= 1 - 1e-3)
        upper_bound = paddle.all((expected_slice / output[..., :3, :3]) <= 1 + 1e-3)

        self.assertTrue(lower_bound and upper_bound)

    @slow
    def test_inference_with_attention(self):
        model = MobileBertModel.from_pretrained("mobilebert-uncased")
        model.eval()
        input_ids = paddle.to_tensor([[101, 7110, 1005, 1056, 2023, 11333, 17413, 1029, 102]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 9, 512]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [2.96605349, 3.73147392, -0.20700839],
                    [2.02441382, 0.04513174, 3.61004543],
                    [4.02399778, -0.25662401, 1.62328660],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
