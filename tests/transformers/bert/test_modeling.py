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
from __future__ import annotations

import unittest
import paddle
from parameterized import parameterized_class

from paddlenlp.transformers import BertModel, BertForQuestionAnswering, BertForSequenceClassification,\
    BertForTokenClassification, BertForPretraining, BertForMultipleChoice, BertForMaskedLM, BertPretrainedModel

from paddlenlp.transformers.bert.configuration import BertConfig

from ..test_modeling_common import ids_tensor, random_attention_mask, ModelTesterMixin, ModelTesterPretrainedMixin
from ...testing_utils import slow

from ..test_configuration_common import ConfigTester


class BertModelTester:

    def __init__(
        self,
        parent: BertModelTest,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
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
        pool_act="tanh",
        fuse=False,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        dropout=0.56,
        return_dict=False,
    ):
        self.parent: unittest.TestCase = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
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
        self.pool_act = pool_act
        self.fuse = fuse
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.dropout = dropout
        self.return_dict = return_dict

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
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size],
                                         self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length],
                                      self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self) -> BertConfig:
        return BertConfig(
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
            pool_act=self.pool_act,
            fuse=self.fuse,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
        )

    def create_and_check_model(
        self,
        config: BertConfig,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model_lambdas = [
            lambda: BertModel(config),
            lambda: BertModel(**config.to_dict()),
        ]
        for model_lambda in model_lambdas:
            model = model_lambda()

            model.eval()
            result = model(input_ids,
                           attention_mask=input_mask,
                           token_type_ids=token_type_ids)
            result = model(input_ids, token_type_ids=token_type_ids)
            result = model(input_ids)
            self.parent.assertEqual(
                result[0].shape,
                [self.batch_size, self.seq_length, self.hidden_size])
            self.parent.assertEqual(result[1].shape,
                                    [self.batch_size, self.hidden_size])

    def create_and_check_for_masked_lm(
        self,
        config: BertConfig,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model_lambdas = [
            lambda: BertForMaskedLM(config),
            lambda: BertForMaskedLM(config=config),
            lambda: BertForMaskedLM(BertModel(config)),
            lambda: BertForMaskedLM(bert=BertModel(config)),
        ]
        for model_lambda in model_lambdas:
            model = model_lambda()
            model.eval()
            result = model(input_ids,
                           attention_mask=input_mask,
                           token_type_ids=token_type_ids,
                           labels=token_labels)
            self.parent.assertEqual(
                result[1].shape,
                [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_model_past_large_inputs(
        self,
        config: BertConfig,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = BertModel(config)
        model.eval()

        # first forward pass
        outputs = model(input_ids,
                        attention_mask=input_mask,
                        use_cache=True,
                        return_dict=self.return_dict)
        past_key_values = outputs.past_key_values if self.return_dict else outputs[
            2]

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), self.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], axis=-1)

        outputs = model(next_input_ids,
                        attention_mask=next_attention_mask,
                        output_hidden_states=True,
                        return_dict=self.return_dict)

        output_from_no_past = outputs[2][0]

        outputs = model(next_tokens,
                        attention_mask=next_attention_mask,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        return_dict=self.return_dict)

        output_from_past = outputs[2][0]

        # select random slice
        random_slice_idx = ids_tensor((1, ), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:,
                                                        random_slice_idx].detach(
                                                        )
        output_from_past_slice = output_from_past[:, :,
                                                  random_slice_idx].detach()

        self.parent.assertTrue(
            output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(
            paddle.allclose(output_from_past_slice,
                            output_from_no_past_slice,
                            atol=1e-3))

    def create_and_check_for_pretraining(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model_lambdas = [
            lambda: BertForPretraining(config),
            lambda: BertForPretraining(config=config),
            lambda: BertForPretraining(BertModel(config)),
            lambda: BertForPretraining(bert=BertModel(config)),
        ]
        for model_lambda in model_lambdas:
            model = model_lambda()
            model.eval()
            result = model(
                input_ids,
                attention_mask=input_mask,
                token_type_ids=token_type_ids,
                labels=token_labels,
                next_sentence_label=sequence_labels,
            )
            self.parent.assertEqual(
                result[1].shape,
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertEqual(result[2].shape, [self.batch_size, 2])

    def create_and_check_for_multiple_choice(
        self,
        config: BertConfig,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model_lambdas = [
            lambda: BertForMultipleChoice(config),
            lambda: BertForMultipleChoice(config, self.num_choices),
            lambda: BertForMultipleChoice(config, self.num_choices, self.dropout
                                          ),
            lambda: BertForMultipleChoice(config=config),
            lambda: BertForMultipleChoice(config=config,
                                          num_choices=self.num_choices),
            lambda: BertForMultipleChoice(config=config,
                                          num_choices=self.num_choices,
                                          dropout=self.dropout),
            lambda: BertForMultipleChoice(BertModel(config)),
            lambda: BertForMultipleChoice(BertModel(config), self.num_choices),
            lambda: BertForMultipleChoice(BertModel(config), self.num_choices,
                                          self.dropout),
            lambda: BertForMultipleChoice(bert=BertModel(config)),
            lambda: BertForMultipleChoice(bert=BertModel(config),
                                          num_choices=self.num_choices),
            lambda: BertForMultipleChoice(bert=BertModel(config),
                                          num_choices=self.num_choices,
                                          dropout=self.dropout),
        ]

        for index, model_lambda in enumerate(model_lambdas):
            model: BertForMultipleChoice = model_lambda()
            model.eval()
            multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(
                [-1, self.num_choices, -1])
            multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(
                [-1, self.num_choices, -1])
            multiple_choice_input_mask = input_mask.unsqueeze(1).expand(
                [-1, self.num_choices, -1])
            result = model(
                multiple_choice_inputs_ids,
                attention_mask=multiple_choice_input_mask,
                token_type_ids=multiple_choice_token_type_ids,
                labels=choice_labels,
            )
            self.parent.assertEqual(result[1].shape,
                                    [self.batch_size, self.num_choices])

            if index % 3 == 2:
                self.parent.assertEqual(model.dropout.p, self.dropout)
            else:
                self.parent.assertEqual(model.dropout.p,
                                        self.hidden_dropout_prob)

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model_lambdas = [
            lambda: BertForQuestionAnswering(config),
            lambda: BertForQuestionAnswering(config, self.dropout),
            lambda: BertForQuestionAnswering(config=config),
            lambda: BertForQuestionAnswering(config=config,
                                             dropout=self.dropout),
            lambda: BertForQuestionAnswering(BertModel(config)),
            lambda: BertForQuestionAnswering(BertModel(config), self.dropout),
            lambda: BertForQuestionAnswering(bert=BertModel(config)),
            lambda: BertForQuestionAnswering(bert=BertModel(config),
                                             dropout=self.dropout)
        ]

        for index, model_lambda in enumerate(model_lambdas):
            model: BertForQuestionAnswering = model_lambda()
            model.eval()
            result = model(
                input_ids,
                attention_mask=input_mask,
                token_type_ids=token_type_ids,
                start_positions=sequence_labels,
                end_positions=sequence_labels,
            )
            self.parent.assertEqual(result[1].shape,
                                    [self.batch_size, self.seq_length])
            self.parent.assertEqual(result[2].shape,
                                    [self.batch_size, self.seq_length])
            if index % 2 == 1:
                self.parent.assertEqual(model.dropout.p, self.dropout)
            else:
                self.parent.assertEqual(model.dropout.p,
                                        self.hidden_dropout_prob)

    def create_and_check_for_sequence_classification(
        self,
        config: BertConfig,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):

        model_lambdas = [
            lambda: BertForSequenceClassification(config),
            lambda: BertForSequenceClassification(config, self.num_labels),
            lambda: BertForSequenceClassification(config, self.num_labels, self.
                                                  dropout),
            lambda: BertForSequenceClassification(config=config),
            lambda: BertForSequenceClassification(config=config,
                                                  num_labels=self.num_labels),
            lambda: BertForSequenceClassification(config=config,
                                                  num_classes=self.num_labels,
                                                  dropout=self.dropout),
            lambda: BertForSequenceClassification(BertModel(config)),
            lambda: BertForSequenceClassification(BertModel(config), self.
                                                  num_labels),
            lambda: BertForSequenceClassification(BertModel(config), self.
                                                  num_labels, self.dropout),
            lambda: BertForSequenceClassification(bert=BertModel(config)),
            lambda: BertForSequenceClassification(bert=BertModel(config),
                                                  num_labels=self.num_labels),
            lambda: BertForSequenceClassification(bert=BertModel(config),
                                                  num_classes=self.num_labels,
                                                  dropout=self.dropout),
        ]
        for index, model_lambda in enumerate(model_lambdas):
            model: BertForSequenceClassification = model_lambda()
            model.eval()
            result = model(input_ids,
                           attention_mask=input_mask,
                           token_type_ids=token_type_ids,
                           labels=sequence_labels)
            self.parent.assertEqual(result[1].shape,
                                    [self.batch_size, self.num_labels])

            if index % 3 == 2:
                self.parent.assertEqual(model.dropout.p, self.dropout)
            else:
                self.parent.assertEqual(model.dropout.p,
                                        self.hidden_dropout_prob)

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model_lambdas = [
            lambda: BertForTokenClassification(config),
            lambda: BertForTokenClassification(config, self.num_labels),
            lambda: BertForTokenClassification(config, self.num_labels, self.
                                               dropout),
            lambda: BertForTokenClassification(config=config),
            lambda: BertForTokenClassification(config=config,
                                               num_labels=self.num_labels),
            lambda: BertForTokenClassification(config=config,
                                               num_classes=self.num_labels,
                                               dropout=self.dropout),
            lambda: BertForTokenClassification(BertModel(config)),
            lambda: BertForTokenClassification(BertModel(config), self.
                                               num_labels),
            lambda: BertForTokenClassification(BertModel(config), self.
                                               num_labels, self.dropout),
            lambda: BertForTokenClassification(bert=BertModel(config)),
            lambda: BertForTokenClassification(bert=BertModel(config),
                                               num_labels=self.num_labels),
            lambda: BertForTokenClassification(bert=BertModel(config),
                                               num_classes=self.num_labels,
                                               dropout=self.dropout),
        ]
        for index, model_lambda in enumerate(model_lambdas):
            model: BertForTokenClassification = model_lambda()

            model.eval()
            result = model(input_ids,
                           attention_mask=input_mask,
                           token_type_ids=token_type_ids,
                           labels=token_labels)
            self.parent.assertEqual(
                result[1].shape,
                [self.batch_size, self.seq_length, self.num_labels])

            if index % 3 == 2:
                self.parent.assertEqual(model.dropout.p, self.dropout)
            else:
                self.parent.assertEqual(model.dropout.p,
                                        self.hidden_dropout_prob)

    def test_addition_params(self, config: BertConfig, *args, **kwargs):
        custom_num_labels, custom_dropout = 7, 0.98

        model_lambdas = [
            lambda: BertForTokenClassification(config, custom_num_labels,
                                               custom_dropout),
            lambda: BertForTokenClassification(config=config,
                                               num_labels=custom_num_labels,
                                               dropout=custom_dropout),
            lambda: BertForTokenClassification(BertModel(
                config), custom_num_labels, custom_dropout),
            lambda: BertForTokenClassification(bert=BertModel(config),
                                               num_labels=custom_num_labels,
                                               dropout=custom_dropout),
        ]
        for model_lambda in model_lambdas:
            model: BertForTokenClassification = model_lambda()
            model.eval()

            self.parent.assertEqual(model.classifier.weight.shape,
                                    [config.hidden_size, custom_num_labels])
            self.parent.assertEqual(model.dropout.p, custom_dropout)

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
            "attention_mask": input_mask
        }
        return config, inputs_dict


@parameterized_class(("return_dict", "use_labels"), [
    [False, False],
    [False, True],
    [True, False],
    [True, True],
])
class BertModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = BertModel
    return_dict = False
    use_labels = False

    all_model_classes = (
        BertModel,
        BertForMaskedLM,
        BertForMultipleChoice,
        BertForPretraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
    )

    def setUp(self):
        super().setUp()

        self.model_tester = BertModelTester(self)
        self.config_tester = ConfigTester(self,
                                          config_class=BertConfig,
                                          vocab_size=256,
                                          hidden_size=24)

    def test_config(self):
        # self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_past_large_inputs(
            *config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(
            *config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(
            *config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(
            *config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(
            *config_and_inputs)

    def test_for_custom_params(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.test_addition_params(*config_and_inputs)

    def test_model_name_list(self):
        config = self.model_tester.get_config()
        model = self.base_model_class(config)
        self.assertTrue(len(model.model_name_list) != 0)


class BertModelPretrainedTest(unittest.TestCase):

    def test_model_config_mapping(self):
        config = BertConfig(num_labels=22, hidden_dropout_prob=0.99)
        self.assertEqual(config.hidden_dropout_prob, 0.99)
        self.assertEqual(config.num_labels, 22)


class BertModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = BertModel

    @slow
    def test_inference_no_attention(self):
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor([[[0.4249, 0.1008, 0.7531],
                                            [0.3771, 0.1188, 0.7467],
                                            [0.4152, 0.1098, 0.7108]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()
        input_ids = paddle.to_tensor(
            [[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor([[[0.4249, 0.1008, 0.7531],
                                            [0.3771, 0.1188, 0.7467],
                                            [0.4152, 0.1098, 0.7108]]])
        self.assertTrue(
            paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
