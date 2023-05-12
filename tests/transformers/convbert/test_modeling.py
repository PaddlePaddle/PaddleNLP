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

from __future__ import annotations

import unittest

import paddle
from parameterized import parameterized_class

from paddlenlp.transformers import (
    ConvBertConfig,
    ConvBertForMaskedLM,
    ConvBertForMultipleChoice,
    ConvBertForPretraining,
    ConvBertForQuestionAnswering,
    ConvBertForSequenceClassification,
    ConvBertForTokenClassification,
    ConvBertModel,
)

from ...testing_utils import slow
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


class ConvBertModelTester:
    def __init__(
        self,
        parent: ConvBertModelTest,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_inputs_embeds=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0,
        embedding_size=16,
        conv_kernel_size=3,
        head_ratio: int = 2,
        num_groups: int = 1,
        pool_act="tanh",
        fuse=False,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        dropout=0.56,
        return_dict=False,
    ):
        self.parent: ConvBertModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_inputs_embeds = use_inputs_embeds
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads // head_ratio
        self.total_num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.embedding_size = embedding_size
        self.conv_kernel_size = conv_kernel_size
        self.head_ratio = head_ratio
        self.num_groups = num_groups
        self.pool_act = pool_act
        self.fuse = fuse
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.dropout = dropout
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        input_ids = None
        inputs_embeds = None
        if self.use_inputs_embeds:
            inputs_embeds = floats_tensor([self.batch_size, self.seq_length, self.embedding_size])
        else:
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
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return (
            config,
            input_ids,
            token_type_ids,
            inputs_embeds,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self) -> ConvBertConfig:
        return ConvBertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.total_num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            embedding_size=self.embedding_size,
            conv_kernel_size=self.conv_kernel_size,
            head_ratio=self.head_ratio,
            num_groups=self.num_groups,
            pool_act=self.pool_act,
            fuse=self.fuse,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
        )

    def create_and_check_model(
        self,
        config: ConvBertConfig,
        input_ids,
        token_type_ids,
        inputs_embeds,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ConvBertModel(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            return_dict=self.return_dict,
        )
        result = model(input_ids, token_type_ids=token_type_ids, return_dict=self.return_dict)
        result = model(input_ids, return_dict=self.return_dict)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        inputs_embeds,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ConvBertForMaskedLM(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            labels=token_labels,
            return_dict=self.return_dict,
        )
        if not self.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_pretraining(
        self,
        config: ConvBertConfig,
        input_ids,
        token_type_ids,
        inputs_embeds,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ConvBertForPretraining(config)
        model.eval()

        generator_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        raw_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            raw_input_ids=raw_input_ids,
            generator_labels=generator_labels,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(result[2].shape, [self.batch_size, self.seq_length])

    def create_and_check_for_multiple_choice(
        self,
        config: ConvBertConfig,
        input_ids,
        token_type_ids,
        inputs_embeds,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ConvBertForMultipleChoice(config)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand([-1, self.num_choices, -1])
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand([-1, self.num_choices, -1])
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            inputs_embeds=inputs_embeds,
            labels=choice_labels,
            return_dict=self.return_dict,
        )

        if not self.return_dict and choice_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if paddle.is_tensor(result):
            result = [result]
        elif choice_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_choices])

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        token_type_ids,
        inputs_embeds,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ConvBertForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            return_dict=self.return_dict,
        )
        if sequence_labels is not None:
            start_logits, end_logits = result[1], result[2]
        else:
            start_logits, end_logits = result[0], result[1]

        self.parent.assertEqual(start_logits.shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(end_logits.shape, [self.batch_size, self.seq_length])

    def create_and_check_for_sequence_classification(
        self,
        config: ConvBertConfig,
        input_ids,
        token_type_ids,
        inputs_embeds,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ConvBertForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            labels=sequence_labels,
            return_dict=self.parent.return_dict,
        )
        if not self.return_dict and sequence_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))
        if sequence_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_labels])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        inputs_embeds,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ConvBertForTokenClassification(config)

        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            labels=token_labels,
            return_dict=self.return_dict,
        )

        if token_labels is not None:
            result = result[1:]
        elif paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.num_labels])

    def test_addition_params(self, config: ConvBertConfig, *args, **kwargs):
        config.num_labels = 7
        config.classifier_dropout = 0.98

        model = ConvBertForTokenClassification(config)
        model.eval()

        self.parent.assertEqual(model.classifier.weight.shape, [config.hidden_size, 7])
        self.parent.assertEqual(model.dropout.p, 0.98)

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
            "inputs_embeds": inputs_embeds,
        }
        return config, inputs_dict


@parameterized_class(
    ("return_dict", "use_labels"),
    [
        [False, False],
        [False, False],
        [False, True],
        [True, False],
        [True, True],
    ],
)
class ConvBertModelTest(ModelTesterMixin, unittest.TestCase):
    test_resize_embeddings: bool = False
    base_model_class = ConvBertModel
    return_dict: bool = False
    use_labels: bool = False
    test_tie_weights: bool = True
    use_test_inputs_embeds: bool = True

    all_model_classes = (
        ConvBertModel,
        ConvBertForMultipleChoice,
        ConvBertForMaskedLM,
        ConvBertForQuestionAnswering,
        ConvBertForSequenceClassification,
        ConvBertForTokenClassification,
    )

    def setUp(self):
        super().setUp()
        self.model_tester = ConvBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ConvBertConfig, vocab_size=256, hidden_size=24)

        self.test_resize_embeddings = False

    def test_config(self):
        # self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_custom_params(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.test_addition_params(*config_and_inputs)

    def test_model_name_list(self):
        config = self.model_tester.get_config()
        model = self.base_model_class(config)
        self.assertTrue(len(model.model_name_list) != 0)

    @slow
    def test_params_compatibility_of_init_method(self):
        """test initing model with different params"""
        model: ConvBertForTokenClassification = ConvBertForTokenClassification.from_pretrained(
            "convbert-base", num_classes=4, dropout=0.3
        )
        assert model.num_labels == 4
        assert model.dropout.p == 0.3


class ConvBertModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):

    base_model_class = ConvBertModel
    paddlehub_remote_test_model_name: str = "convbert-base"

    @slow
    def test_inference_no_attention(self):
        model = ConvBertModel.from_pretrained("convbert-base")
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 6, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[-0.0864, -0.4898, -0.3677], [0.1434, -0.2952, -0.7640], [-0.0112, -0.4432, -0.5432]]]
        )
        self.assertTrue(paddle.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @unittest.skip(
        "The URL of CONVBERT_PRETRAINED_RESOURCE_FILES_MAP in configuration.py is not in the format required by test_pretrained_save_and_load"
    )
    def test_pretrained_save_and_load(self):
        pass


if __name__ == "__main__":
    unittest.main()
