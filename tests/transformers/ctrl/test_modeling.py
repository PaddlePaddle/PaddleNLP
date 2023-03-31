# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
    CTRLConfig,
    CTRLForSequenceClassification,
    CTRLLMHeadModel,
    CTRLModel,
    CTRLPreTrainedModel,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


class CTRLModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_token_type_ids=True,
        use_input_mask=True,
        use_labels=True,
        use_mc_token_ids=True,
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
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_token_type_ids = use_token_type_ids
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.use_mc_token_ids = use_mc_token_ids
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
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.pad_token_id = self.vocab_size - 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        mc_token_ids = None
        if self.use_mc_token_ids:
            mc_token_ids = ids_tensor([self.batch_size, self.num_choices], self.seq_length)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return CTRLConfig(
            vocab_size=self.vocab_size,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_ctrl_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = CTRLModel(config=config)
        model.eval()

        model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = CTRLLMHeadModel(config)
        model.eval()

        result = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)
        self.parent.assertEqual(result[0].shape, [1])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}

        return config, inputs_dict

    def create_and_check_ctrl_for_sequence_classification(
        self, config, input_ids, input_mask, head_mask, token_type_ids, *args
    ):
        config.num_classes = self.num_labels
        model = CTRLForSequenceClassification(config)
        model.eval()
        sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        result = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(len(result), 2)
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.num_labels])


@parameterized_class(
    ("use_labels"),
    [
        [False],
        [True],
    ],
)
class CTRLModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = CTRLModel
    return_dict = False
    use_labels = False
    all_model_classes = (CTRLModel, CTRLLMHeadModel, CTRLForSequenceClassification)

    def setUp(self):
        super().setUp()
        self.model_tester = CTRLModelTester(self)

    def test_ctrl_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_ctrl_model(*config_and_inputs)

    def test_ctrl_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_ctrl_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_ctrl_for_sequence_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(CTRLPreTrainedModel.pretrained_init_configuration)[:1]:
            model = CTRLModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class CTRLModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_ctrl(self):
        model = CTRLLMHeadModel.from_pretrained("ctrl")
        input_ids = paddle.to_tensor(
            [[11859, 0, 1611, 8]],
        )  # Legal the president is
        expected_output_ids = [
            11859,
            0,
            1611,
            8,
            5,
            150,
            26449,
            2,
            19,
            348,
            469,
            3,
            2595,
            48,
            20740,
            246533,
            246533,
            19,
            30,
            5,
        ]  # Legal the president is a good guy and I don't want to lose my job. \n \n I have a

        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
