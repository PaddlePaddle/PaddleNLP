# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 MiniMax AI. All rights reserved.
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

import gc
import unittest

import paddle

from paddlenlp.transformers import (
    MiniMaxText01Config,
    MiniMaxText01ForCausalLM,
    MiniMaxText01ForSequenceClassification,
)
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)

from ...testing_utils import require_gpu


class MiniMaxText01ModelTester:
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
        hidden_size=99,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=1e6,
        sliding_window=32,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=2,
        rms_norm_eps=1e-5,
        scope=None,
        attn_type_list=["0", "1"],
    ):
        self.parent: MiniMaxText01ModelTest = parent
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
        self.num_key_value_heads = num_key_value_heads
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
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.rms_norm_eps = rms_norm_eps
        self.scope = scope
        self.attn_type_list = attn_type_list

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

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

    def get_config(self) -> MiniMaxText01Config:
        return MiniMaxText01Config(
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
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            rope_theta=self.rope_theta,
            sliding_window=self.sliding_window,
            attention_dropout=self.attention_dropout,
            num_experts_per_tok=self.num_experts_per_tok,
            num_local_experts=self.num_local_experts,
            rms_norm_eps=self.rms_norm_eps,
            attn_type_list=self.attn_type_list,
        )

    def create_and_check_model(
        self,
        config: MiniMaxText01Config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = MiniMaxText01ForCausalLM(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        model = MiniMaxText01ForCausalLM(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

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
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class MiniMaxText01ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = None
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (MiniMaxText01ForCausalLM, MiniMaxText01ForSequenceClassification)
    all_generative_model_classes = {MiniMaxText01ForCausalLM: {None, "minimax_text01"}}
    pipeline_model_mapping = {
        "text-classification": MiniMaxText01ForSequenceClassification,
        "text-generation": MiniMaxText01ForCausalLM,
        "zero-shot": MiniMaxText01ForSequenceClassification,
    }

    def setUp(self):
        super().setUp()
        self.model_tester = MiniMaxText01ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MiniMaxText01Config, hidden_size=768)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_MiniMaxText01_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = MiniMaxText01ForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])

    def test_MiniMaxText01_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = MiniMaxText01ForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])

    def test_MiniMaxText01_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(paddle.float32)
        model = MiniMaxText01ForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])


class MiniMaxText01IntegrationTest(unittest.TestCase):
    @require_gpu(1)
    def test_model_tiny_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = MiniMaxText01ForCausalLM.from_pretrained(
            "__internal_testing__/MiniMax-Text-01-l2-tiny-pd", dtype="float32"
        )
        model.eval()
        input_ids = paddle.to_tensor([input_ids])
        with paddle.no_grad():
            out = model(input_ids, return_dict=True).logits
        EXPECTED_MEAN = paddle.to_tensor(
            [[-0.00203404, 0.00172575, 0.00171089, 0.00109741, -0.00046862, 0.00017896, -0.00002699, -0.00206279]]
        )
        paddle.allclose(out.mean(-1), EXPECTED_MEAN, atol=1e-6, rtol=1e-6)
        EXPECTED_SLICE = paddle.to_tensor(
            [
                -0.45604241,
                0.44674566,
                0.01559911,
                -0.22750290,
                0.46994418,
                -0.39009440,
                -0.58710217,
                -0.65201938,
                1.06324077,
                0.28406841,
                0.22498111,
                0.36873919,
                0.22047190,
                -0.47585970,
                -0.16434811,
                0.20234424,
                -0.32718620,
                0.32738528,
                0.36627784,
                -0.76008093,
                -0.15530412,
                0.63310510,
                0.49225768,
                0.57552850,
                -0.15108462,
                -0.71018273,
                0.11868254,
                -0.06228763,
                0.08378446,
                -0.84608293,
            ]
        )
        paddle.allclose(out[0, 0, :30], EXPECTED_SLICE, atol=1e-6, rtol=1e-6)

        del model
        paddle.device.cuda.empty_cache()
        gc.collect()
