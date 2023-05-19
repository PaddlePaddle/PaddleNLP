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
from typing import Any, Dict, Tuple

import paddle
from paddle import Tensor

from paddlenlp.transformers import (
    ErnieGramConfig,
    ErnieGramForQuestionAnswering,
    ErnieGramForSequenceClassification,
    ErnieGramForTokenClassification,
    ErnieGramModel,
    ErnieGramPretrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


class ErnieGramModelTester:
    """Base ErnieGram Model tester which can test:"""

    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.seq_length = 7
        self.is_training = False
        self.use_token_type_ids = True
        self.use_attention_mask = True
        self.test_resize_embeddings = False

        self.num_labels = 3
        self.attention_probs_dropout_prob = 0.1
        self.embedding_size = 8
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 8
        self.initializer_range = 0.02
        self.max_position_embeddings = 512
        self.num_attention_heads = 2
        self.num_hidden_layers = 2
        self.type_vocab_size = 2
        self.vocab_size = 1801
        self.config = ErnieGramConfig(
            num_labels=self.num_labels,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            embedding_size=self.embedding_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_size=self.hidden_size,
            initializer_range=self.initializer_range,
            max_position_embeddings=self.max_position_embeddings,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            type_vocab_size=self.type_vocab_size,
            vocab_size=self.vocab_size,
        )

    def prepare_config_and_inputs(self) -> Tuple[Dict[str, Any], Tensor, Tensor, Tensor]:
        config = self.config
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = paddle.zeros_like(input_ids)

        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)
        return config, input_ids, token_type_ids, attention_mask, sequence_labels, token_labels, choice_labels

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, token_type_ids, attention_mask, _, _, _ = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = ErnieGramModel(config)
        model.eval()

        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=self.parent.return_dict,
        )
        if paddle.is_tensor(result):
            result = [result]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.config.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.config.hidden_size])

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = ErnieGramForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=sequence_labels,
            return_dict=self.parent.return_dict,
        )
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))

        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.config.num_labels])

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = ErnieGramForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            start_position=sequence_labels,
            end_position=sequence_labels,
            return_dict=self.parent.return_dict,
        )
        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length])

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        sequence_labels: Tensor,
        token_labels: Tensor,
        choice_labels: Tensor,
    ):
        model = ErnieGramForTokenClassification(config)
        model.eval()
        result = model(
            input_ids,
            token_type_ids=token_type_ids,
            labels=token_labels,
            return_dict=self.parent.return_dict,
            attention_mask=attention_mask,
        )
        if not self.parent.return_dict and token_labels is None:
            self.parent.assertTrue(paddle.is_tensor(result))
        if paddle.is_tensor(result):
            result = [result]
        elif token_labels is not None:
            result = result[1:]

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.config.num_labels])

    def create_and_check_model_cache(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = ErnieGramModel(config)
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
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-4))

    def get_config(self) -> dict:
        return self.config


class ErnieGramModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = ErnieGramModel
    return_dict = False
    use_labels = False

    all_model_classes = (
        ErnieGramModel,
        ErnieGramForSequenceClassification,
        ErnieGramForTokenClassification,
        ErnieGramForQuestionAnswering,
    )

    def setUp(self):
        self.model_tester = ErnieGramModelTester(self)
        self.test_resize_embeddings = self.model_tester.test_resize_embeddings

    def get_config():
        pass

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

    def test_for_model_cache(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_cache(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(ErnieGramPretrainedModel.pretrained_init_configuration)[:1]:
            model = ErnieGramModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class ErnieGramModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_attention(self):
        model = ErnieGramModel.from_pretrained("ernie-gram-zh")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [-0.43569842, -1.50805628, -2.24448967],
                    [-0.12123521, -1.35024536, -1.76512492],
                    [-0.14853711, -1.13618660, -2.87098265],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-5))

    @slow
    def test_inference_with_attention(self):
        model = ErnieGramModel.from_pretrained("ernie-gram-zh-finetuned-dureader-robust")
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
                    [0.37543082, -2.94639230, -2.04799986],
                    [0.14168003, -2.02873731, -2.34919119],
                    [0.70280838, -2.40280604, -1.93488157],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_past_key_value(self):
        model = ErnieGramModel.from_pretrained("ernie-gram-zh")
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
                    [-0.43569842, -1.50805628, -2.24448967],
                    [-0.12123521, -1.35024536, -1.76512492],
                    [-0.14853711, -1.13618660, -2.87098265],
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
                    [-0.59400421, -1.32317221, -2.88611341],
                    [-0.79759967, -0.97396499, -1.89245439],
                    [-0.47301087, -1.50476563, -2.37942648],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[0][:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
