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

import inspect
import unittest

import paddle

from paddlenlp.transformers import (
    FunnelConfig,
    FunnelForQuestionAnswering,
    FunnelForSequenceClassification,
    FunnelForTokenClassification,
    FunnelModel,
)

from ..test_modeling_common import ModelTesterMixin, ids_tensor


class FunnelModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        vocab_size=99,
        block_sizes=[4, 4, 4],
        block_repeats=None,
        num_decoder_layers=2,
        d_model=32,
        n_head=4,
        d_head=4,
        d_inner=32,
        hidden_act="gelu_new",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        max_position_embeddings=512,
        type_vocab_size=3,
        initializer_range=0.1,
        initializer_std=None,
        layer_norm_eps=1e-9,
        num_labels=2,
        return_dict=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.vocab_size = vocab_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.block_sizes = block_sizes
        self.block_repeats = block_repeats
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.hidden_size = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.initializer_std = initializer_std
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = sum(self.block_sizes)
        self.num_attention_heads = n_head
        self.num_labels = num_labels
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = paddle.ones([self.batch_size, self.seq_length], dtype="int32")

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        config = self.get_config()
        return_dict = self.return_dict
        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            return_dict,
        )

    def get_config(self):
        return FunnelConfig(
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            block_sizes=self.block_sizes,
            block_repeats=self.block_repeats,
            num_decoder_layers=self.num_decoder_layers,
            d_model=self.d_model,
            n_head=self.n_head,
            d_head=self.d_head,
            d_inner=self.d_inner,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            initializer_std=self.initializer_std,
            layer_norm_eps=self.layer_norm_eps,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            return_dict,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "return_dict": return_dict,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        return_dict,
    ):
        model = FunnelModel(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.d_model])

    def create_and_check_question_answering(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        return_dict,
    ):
        model = FunnelForQuestionAnswering(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length])

    def create_and_check_sequence_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        return_dict,
    ):
        model = FunnelForSequenceClassification(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.num_labels])

    def create_and_check_token_classification(self, config, input_ids, token_type_ids, input_mask, return_dict):
        model = FunnelForTokenClassification(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.num_labels])


class FunnelModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = FunnelModel
    return_dict: bool = True
    use_labels: bool = False
    use_test_inputs_embeds: bool = False

    all_model_classes = (FunnelModel,)

    def setUp(self):
        self.model_tester = FunnelModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_question_answering(*config_and_inputs)

    def test_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_sequence_classification(*config_and_inputs)

    def test_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_token_classification(*config_and_inputs)

    def test_attention_outputs(self):
        "attention include encoder and decoder"
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            signature = inspect.signature(model_class.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            if not all(name in arg_names for name in ["output_attentions", "output_hidden_states", "return_dict"]):
                continue
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            inputs_dict["return_dict"] = True
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if self.is_encoder_decoder else outputs.attentions
            self.assertEqual(
                len(attentions), self.model_tester.num_hidden_layers + self.model_tester.num_decoder_layers
            )

            # TODO(guosheng): check that output_attentions also work using config
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # loss is at first position
                if "labels" in inputs_dict:
                    correct_outlen += 1  # loss is added to beginning
                # Question Answering model returns start_logits and end_logits
                if model_class.__name__.endswith("ForQuestionAnswering"):
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output
                if "past_key_values" in outputs:
                    correct_outlen += 1  # past_key_values have been returned

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1

            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if self.is_encoder_decoder else outputs.attentions

            self.assertEqual(
                len(self_attentions), self.model_tester.num_hidden_layers + self.model_tester.num_decoder_layers
            )

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    def test_hidden_states_output(self):
        "hidden state include encoder and decoder"

        def check_hidden_states_output(inputs_dict, config, model_class):
            model = self._make_model_instance(config, model_class)
            model.eval()

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if self.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester,
                "expected_num_hidden_layers",
                self.model_tester.num_hidden_layers + 1 + self.model_tester.num_decoder_layers + 1,
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if self.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict["return_dict"] = True
        for model_class in self.all_model_classes:
            signature = inspect.signature(model_class.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            if not all(name in arg_names for name in ["output_attentions", "output_hidden_states", "return_dict"]):
                continue
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)
            # TODO(guosheng): check that output_hidden_states also work using config
