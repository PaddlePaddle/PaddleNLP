# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
    GPTJConfig,
    GPTJForCausalLM,
    GPTJForQuestionAnswering,
    GPTJForSequenceClassification,
    GPTJModel,
)

from ...testing_utils import slow
from ..test_generation_utils import GenerationTesterMixin
from ..test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)

GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST = ["EleutherAI/gpt-j-6B"]


class GPTJModelTester:
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
        rotary_dim=4,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
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
        self.rotary_dim = rotary_dim
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
        self.scope = None
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

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

        return (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return GPTJConfig(
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
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            rotary_dim=self.rotary_dim,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_gptj_model(self, config, input_ids, input_mask, token_type_ids, *args):
        model = GPTJModel(config=config)
        model.eval()
        result = model(input_ids, token_type_ids=token_type_ids, use_cache=True, return_dict=True)
        result = model(input_ids, token_type_ids=token_type_ids, use_cache=True, return_dict=True)
        result = model(input_ids, use_cache=True, return_dict=True)

        self.parent.assertEqual(
            result.last_hidden_state.shape, list((self.batch_size, self.seq_length, self.hidden_size))
        )
        self.parent.assertEqual(len(result.past_key_values), config.n_layer)

    def create_and_check_gptj_model_past(self, config, input_ids, input_mask, token_type_ids, *args):
        model = GPTJModel(config=config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, token_type_ids=token_type_ids, use_cache=True)
        outputs_no_past = model(input_ids, token_type_ids=token_type_ids, use_cache=False)
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past = outputs

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
        next_token_types = ids_tensor([self.batch_size, 1], self.type_vocab_size)

        # append to next input_ids and token_type_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_token_type_ids = paddle.concat([token_type_ids, next_token_types], axis=-1)

        output_from_no_past = model(next_input_ids, token_type_ids=next_token_type_ids, return_dict=True)[
            "last_hidden_state"
        ]
        output_from_past = model(next_tokens, token_type_ids=next_token_types, past_key_values=past, return_dict=True)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=5e-1))

    def create_and_check_gptj_model_attention_mask_past(self, config, input_ids, input_mask, token_type_ids, *args):
        model = GPTJModel(config=config)
        model.eval()

        # create attention mask
        attn_mask = paddle.ones(input_ids.shape, dtype=paddle.int64)
        half_seq_length = self.seq_length // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past = model(input_ids, attention_mask=attn_mask, use_cache=True)

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        attn_mask = paddle.concat(
            [attn_mask, paddle.ones((attn_mask.shape[0], 1), dtype=paddle.int64)],
            axis=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask, return_dict=True)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past, attention_mask=attn_mask, return_dict=True)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_gptj_model_past_large_inputs(self, config, input_ids, input_mask, token_type_ids, *args):
        model = GPTJModel(config=config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask, use_cache=True)

        output, past = outputs

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_token_types = ids_tensor([self.batch_size, 3], self.type_vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and token_type_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_token_type_ids = paddle.concat([token_type_ids, next_token_types], axis=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], axis=-1)
        output_from_no_past = model(
            next_input_ids, token_type_ids=next_token_type_ids, attention_mask=next_attention_mask, return_dict=True
        )["last_hidden_state"]
        output_from_past = model(
            next_tokens,
            token_type_ids=next_token_types,
            attention_mask=next_attention_mask,
            past_key_values=past,
            return_dict=True,
        )["last_hidden_state"]
        self.parent.assertTrue(output_from_past.shape[1] == next_tokens.shape[1])

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, token_type_ids, *args):
        model = GPTJForCausalLM(config)
        model.eval()

        result = model(input_ids, token_type_ids=token_type_ids, labels=input_ids, return_dict=True)
        self.parent.assertIsInstance(result.loss.item(), float)
        self.parent.assertEqual(result.logits.shape, list((self.batch_size, self.seq_length, self.vocab_size)))

    def create_and_check_forward_and_backwards(
        self, config, input_ids, input_mask, token_type_ids, *args, gradient_checkpointing=False
    ):
        model = GPTJForCausalLM(config)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        result = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        result.loss.backward()

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids.astype("int64"),
            "token_type_ids": token_type_ids,
        }

        return config, inputs_dict


class GPTJModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = GPTJModel

    all_model_classes = (GPTJModel, GPTJForCausalLM, GPTJForSequenceClassification, GPTJForQuestionAnswering)
    all_generative_model_classes = {GPTJForCausalLM: (GPTJModel, "unimo")}
    fx_compatible = True
    test_pruning = False
    test_missing_keys = False
    test_model_parallel = False
    test_head_masking = False
    use_test_model_name_list = False

    # special case for DoubleHeads model
    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class)
        return inputs_dict

    def setUp(self):
        self.model_tester = GPTJModelTester(self)

    def test_gptj_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gptj_model(*config_and_inputs)

    def test_gptj_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gptj_model_past(*config_and_inputs)

    def test_gptj_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gptj_model_attention_mask_past(*config_and_inputs)

    def test_gptj_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gptj_model_past_large_inputs(*config_and_inputs)

    def test_gptj_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = GPTJModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
