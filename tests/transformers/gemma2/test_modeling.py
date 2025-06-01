# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma2Model,
)

from ...testing_utils import slow
from ..test_modeling_common import ModelTesterMixin, ids_tensor


class Gemma2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_key_value_heads=2,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
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
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = paddle.ones([self.batch_size, self.seq_length], dtype="int64")

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], 2)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels

    def get_config(self):
        return Gemma2Config(
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
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            num_key_value_heads=self.num_key_value_heads,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels):
        model = Gemma2Model(config)
        model.eval()

        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(self, config, input_ids, input_mask, sequence_labels, token_labels):
        model = Gemma2ForCausalLM(config)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, sequence_labels, token_labels) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class Gemma2ModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = Gemma2Model
    all_model_classes = (Gemma2Model, Gemma2ForCausalLM)
    all_generative_model_classes = {Gemma2ForCausalLM: (Gemma2Model, "gemma2")}
    test_resize_embeddings = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = Gemma2ModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in ["gemma2-2b", "gemma2-7b"]:
            model = Gemma2Model.from_pretrained(model_name)
            self.assertIsNotNone(model)
