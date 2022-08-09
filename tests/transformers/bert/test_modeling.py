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

from paddlenlp.transformers import BertModel, BertForQuestionAnswering, BertForSequenceClassification,\
    BertForTokenClassification, BertForPretraining, BertForMultipleChoice, BertForMaskedLM, BertPretrainedModel
from tests.transformers.test_modeling_common import ids_tensor, floats_tensor, random_attention_mask, ModelTesterMixin
from tests.testing_utils import slow


class BertModelTester:

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
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0,
        pool_act="tanh",
        fuse=False,
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
        self.pool_act = pool_act
        self.fuse = fuse
        self.type_sequence_label_size = type_sequence_label_size
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

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

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask

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
            "pad_token_id": self.pad_token_id,
            "pool_act": self.pool_act,
            "fuse": self.fuse,
        }

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = BertModel(**config)
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
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = BertForMaskedLM(BertModel(**config))
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids)
        self.parent.assertEqual(
            result.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = BertForMultipleChoice(BertModel(**config),
                                      num_choices=self.num_choices)
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
        )
        self.parent.assertEqual(result.shape,
                                [self.batch_size, self.num_choices])

    def create_and_check_for_question_answering(self, config, input_ids,
                                                token_type_ids, input_mask):
        model = BertForQuestionAnswering(BertModel(**config))
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(result[0].shape,
                                [self.batch_size, self.seq_length])
        self.parent.assertEqual(result[1].shape,
                                [self.batch_size, self.seq_length])

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = BertForSequenceClassification(BertModel(**config),
                                              num_classes=self.num_classes)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(result.shape,
                                [self.batch_size, self.num_classes])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
    ):
        model = BertForTokenClassification(BertModel(**config),
                                           num_classes=self.num_classes)
        model.eval()
        result = model(input_ids,
                       attention_mask=input_mask,
                       token_type_ids=token_type_ids)
        self.parent.assertEqual(
            result.shape, [self.batch_size, self.seq_length, self.num_classes])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask
        }
        return config, inputs_dict


class BertModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = BertModel

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
        self.model_tester = BertModelTester(self)

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

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(
            *config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(
                BertPretrainedModel.pretrained_init_configuration)[:1]:
            model = BertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class BertModelIntegrationTest(unittest.TestCase):

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
