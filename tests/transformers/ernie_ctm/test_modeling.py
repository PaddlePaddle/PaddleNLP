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

from paddlenlp.transformers import (
    ErnieCtmConfig,
    ErnieCtmForTokenClassification,
    ErnieCtmModel,
    ErnieCtmNptagModel,
    ErnieCtmWordtagModel,
)

from ...testing_utils import slow
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    ids_tensor,
    random_attention_mask,
)


class ErnieCtmModelTester:
    def __init__(
        self,
        parent: ErnieCtmModelTest,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size: int = 100,
        embedding_size: int = 16,
        hidden_size: int = 16,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        intermediate_size: int = 16,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        use_content_summary: bool = True,
        content_summary_index: int = 1,
        cls_num: int = 2,
        pad_token_id: int = 0,
        num_prompt_placeholders: int = 5,
        prompt_vocab_ids: set = None,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        dropout=0.56,
        return_dict=False,
    ):
        self.parent: ErnieCtmModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.use_content_summary = use_content_summary
        self.content_summary_index = content_summary_index
        self.cls_num = cls_num
        self.pad_token_id = pad_token_id
        self.num_prompt_placeholders = num_prompt_placeholders
        self.prompt_vocab_ids = prompt_vocab_ids
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.dropout = dropout
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
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
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self) -> ErnieCtmConfig:
        return ErnieCtmConfig(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            layer_norm_eps=self.layer_norm_eps,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            use_content_summary=self.use_content_summary,
            content_summary_index=self.content_summary_index,
            cls_num=self.cls_num,
            pad_token_id=self.pad_token_id,
            num_prompt_placeholders=self.num_prompt_placeholders,
            prompt_vocab_ids=self.prompt_vocab_ids,
            type_sequence_label_size=self.type_sequence_label_size,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
            scope=self.scope,
            dropout=self.dropout,
            return_dict=self.return_dict,
        )

    def create_and_check_model(
        self,
        config: ErnieCtmConfig,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ErnieCtmModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.hidden_size])

    def create_and_check_model_past_large_inputs(
        self,
        config: ErnieCtmConfig,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ErnieCtmModel(config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True, return_dict=self.return_dict)
        past_key_values = outputs.past_key_values if self.return_dict else outputs[2]

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), self.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], axis=-1)

        outputs = model(
            next_input_ids, attention_mask=next_attention_mask, output_hidden_states=True, return_dict=self.return_dict
        )

        output_from_no_past = outputs[2][0]

        outputs = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=self.return_dict,
        )

        output_from_past = outputs[2][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

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
        model = ErnieCtmForTokenClassification(config)

        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.num_labels])

    def create_and_check_for_wordtag(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ErnieCtmWordtagModel(config)

        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.num_labels])

    def create_and_check_for_nptag(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = ErnieCtmNptagModel(config)

        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])

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
class ErnieCtmModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = ErnieCtmModel
    return_dict = False
    use_labels = False
    is_encoder_decoder = False

    all_model_classes = (
        ErnieCtmModel,
        ErnieCtmWordtagModel,
        ErnieCtmNptagModel,
        ErnieCtmForTokenClassification,
    )

    def setUp(self):
        super().setUp()

        self.model_tester = ErnieCtmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ErnieCtmConfig, vocab_size=256, hidden_size=24)

    def test_config(self):
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_wordtag(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_wordtag(*config_and_inputs)

    def test_for_nptag(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_nptag(*config_and_inputs)

    def test_model_name_list(self):
        config = self.model_tester.get_config()
        model = self.base_model_class(config)
        self.assertTrue(len(model.model_name_list) != 0)


class ErnieCtmModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = ErnieCtmModel
    paddlehub_remote_test_model_path = "__internal_testing__/tiny-random-ernie_ctm"

    @slow
    def test_inference_no_attention(self):
        model = ErnieCtmModel.from_pretrained(self.paddlehub_remote_test_model_path)
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[0.4249, 0.1008, 0.7531], [0.3771, 0.1188, 0.7467], [0.4152, 0.1098, 0.7108]]]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = ErnieCtmModel.from_pretrained(self.paddlehub_remote_test_model_path)
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [[[0.4249, 0.1008, 0.7531], [0.3771, 0.1188, 0.7467], [0.4152, 0.1098, 0.7108]]]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
