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

from paddlenlp.transformers import (
    Phi3Config,
    Phi3Model,
    Phi3ForCausalLM,
)

from ..test_modeling_common import ids_tensor, ModelTesterMixin, GenerationTesterMixin


class Phi3ModelTester:
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
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_labels=3,
        scope=None,
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
        self.num_key_value_heads = num_key_value_heads
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
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = paddle.ones([self.batch_size, self.seq_length], dtype="int64")
            input_mask = paddle.where(
                input_ids == self.pad_token_id,
                paddle.zeros_like(input_mask),
                input_mask,
            )

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.num_labels)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels

    def get_config(self):
        return Phi3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
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
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
    ):
        model = Phi3Model(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
    ):
        model = Phi3ForCausalLM(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            labels=token_labels,
        )
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class Phi3ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Phi3Model
    return_dict = False
    use_labels = False
    test_tie_weights = True

    all_model_classes = (
        Phi3Model,
        Phi3ForCausalLM,
    )
    all_generative_model_classes = {Phi3ForCausalLM: (Phi3Model, "phi3")}

    def setUp(self):
        self.model_tester = Phi3ModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)


if __name__ == "__main__":
    unittest.main()
