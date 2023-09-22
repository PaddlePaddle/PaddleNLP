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

import math
import random
import unittest

import numpy as np
import paddle
from parameterized import parameterized_class

from paddlenlp.transformers import GPTTokenizer, OPTConfig, OPTForCausalLM, OPTModel
from tests.testing_utils import PaddleNLPModelTest, require_package, slow
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (  # GenerationD2STestMixin,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
]


class OPTModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        normalize_before=True,
        word_embed_proj_dim=32,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
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
        self.scope = None
        self.normalize_before = normalize_before
        self.word_embed_proj_dim = word_embed_proj_dim
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")

        input_mask = None
        if self.use_input_mask:
            # contruct input_mask filling with 0 and -1e4
            # left padding: [[-1e4, -1e4, -1e4, 0, 0], [-1e4, -1e4, -1e4, 0, 0], ...]
            input_mask = []
            for _ in range(self.batch_size):
                pad_length = random.randint(0, self.seq_length)
                input_mask.append([0] * (self.seq_length - pad_length) + [1] * pad_length)
            input_mask = paddle.to_tensor(input_mask, dtype="int64")

        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size, dtype="int64")
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels, dtype="int64")
            choice_labels = ids_tensor([self.batch_size], self.num_choices, dtype="int64")

        config = self.get_config()

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return OPTConfig(
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
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            normalize_before=self.normalize_before,
            word_embed_proj_dim=self.word_embed_proj_dim,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = paddle.cast(
            ids_tensor([self.batch_size, self.seq_length], vocab_size=2), dtype="int64"
        )

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_opt_model(self, config, input_ids, input_mask, *args):
        model = OPTModel(config)
        model.eval()

        result = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)
        result = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)
        result = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)

        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(len(result[1]), config.num_hidden_layers)

    def create_and_check_opt_model_past(self, config, input_ids, input_mask, *args):
        model = OPTModel(config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)
        outputs_use_cache_conf = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)
        model(input_ids, use_cache=False, return_dict=self.parent.return_dict)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))

        output, past = outputs[:2]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size, dtype="int64")

        # append to next input_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids, return_dict=self.parent.return_dict)
        if self.parent.return_dict:
            output_from_no_past = output_from_no_past[0]

        past_key_values_length = paddle.shape(past[0].k)[2]
        attention_mask = paddle.ones(shape=[next_tokens.shape[0], 1 + past_key_values_length])
        output_from_past = model(
            next_tokens, use_cache=True, attention_mask=attention_mask, cache=past, return_dict=self.parent.return_dict
        )[0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_opt_model_past_large_inputs(self, config, input_ids, input_mask, *args):
        model = OPTModel(config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True, return_dict=self.parent.return_dict)
        output, past = outputs[:2]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size, dtype="int64")
        next_mask = paddle.ones_like(next_tokens, dtype=paddle.int64)

        # append to next input_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], axis=-1)

        output_from_no_past = model(
            next_input_ids, attention_mask=next_attention_mask, return_dict=self.parent.return_dict
        )
        if self.parent.return_dict:
            output_from_no_past = output_from_no_past[0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            cache=past,
            use_cache=True,
            return_dict=self.parent.return_dict,
        )[0]
        self.parent.assertTrue(output_from_past.shape[1] == next_tokens.shape[1])

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1], dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_opt_for_causal_lm(self, config, input_ids, input_mask, *args):
        model = OPTForCausalLM(config)
        model.eval()

        result = model(
            input_ids,
            use_cache=True,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )

        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_forward_and_backwards(self, config, input_ids, input_mask, *args):
        model = OPTForCausalLM(config)

        if self.parent.use_labels:
            loss, logits = model(input_ids, labels=input_ids, return_dict=self.parent.return_dict)
            self.parent.assertEqual(loss.shape, [1])
            self.parent.assertEqual(logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
            loss.backward()

    def create_and_check_opt_weight_initialization(self, config, *args):
        model = OPTModel(config)
        model_std = model.config.initializer_range / math.sqrt(2 * model.config.num_hidden_layers)
        for key in model.state_dict().keys():
            if "out_proj" in key and "weight" in key:
                self.parent.assertLessEqual(abs((paddle.std(model.state_dict()[key]) - model_std).numpy()), 0.02)
                self.parent.assertLessEqual(abs((paddle.mean(model.state_dict()[key]) - 0.0).numpy()), 0.01)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
        }

        return config, inputs_dict

    def create_and_check_model_cache(self, config, input_ids, input_mask, *args):
        model = OPTModel(config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True, return_dict=self.parent.return_dict)
        past_key_values = outputs.past_key_values if self.parent.return_dict else outputs[1]

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), self.vocab_size, dtype="int64")

        # all next mask is ones
        next_mask = paddle.ones_like(next_tokens, dtype="int64")

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], axis=-1)

        outputs = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            output_hidden_states=True,
            return_dict=self.parent.return_dict,
        )

        output_from_no_past = outputs[1][0]

        outputs = model(
            next_tokens,
            attention_mask=next_attention_mask,
            cache=past_key_values,
            output_hidden_states=True,
            return_dict=self.parent.return_dict,
        )

        output_from_past = outputs[1][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))


@parameterized_class(
    ("return_dict", "use_labels"),
    [
        [False, False],
        [False, True],
        [True, False],
        [True, True],
    ],
)
class OPTModelTest(ModelTesterMixin, GenerationTesterMixin, PaddleNLPModelTest):
    base_model_class = OPTModel
    use_labels = False
    return_dict = False
    use_test_inputs_embeds = True

    all_model_classes = [
        OPTModel,
    ]
    all_generative_model_classes = {OPTForCausalLM: (OPTModel, "opt")}
    test_missing_keys = False
    test_model_parallel = True

    # special case for DoubleHeads model
    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class)

        return inputs_dict

    def setUp(self):
        self.model_tester = OPTModelTester(self)
        random.seed(128)
        np.random.seed(128)
        paddle.seed(128)

    def test_opt_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_opt_model(*config_and_inputs)

    def test_opt_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_opt_model_past(*config_and_inputs)

    def test_opt_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_opt_model_past_large_inputs(*config_and_inputs)

    def test_opt_causal_lm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_opt_for_causal_lm(*config_and_inputs)

    def test_opt_weight_initialization(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_opt_weight_initialization(*config_and_inputs)

    def test_for_model_cache(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_cache(*config_and_inputs)

    @slow
    def test_batch_generation(self):
        model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b")
        model.eval()
        tokenizer = GPTTokenizer.from_pretrained("facebook/opt-1.3b")
        tokenizer.padding_side = "left"

        # use different length sentences to test batching
        sentences = [
            "my dog is",
            "Today, I",
        ]

        inputs = tokenizer(sentences, return_tensors="pd", padding=True)
        input_ids = inputs["input_ids"]

        outputs, _ = model.generate(
            input_ids=input_ids,
            decode_strategy="greedy_search",
            use_cache=True,
        )
        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pd")["input_ids"]
        output_non_padded, _ = model.generate(
            input_ids=inputs_non_padded, use_cache=True, decode_strategy="greedy_search"
        )
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)

        inputs_padded = tokenizer(sentences[1], return_tensors="pd")["input_ids"]
        output_padded, _ = model.generate(input_ids=inputs_padded, use_cache=True, decode_strategy="greedy_search")
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            " a rescue and she's the best dog ever. she's a little bitch but she's the best",
            " am going to share with you a few of my favorite recipes.\nI have been cooking for a",
        ]

        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_ids = inputs_dict[self.input_name]

        max_batch_size = 2
        sequence_length = input_ids.shape[-1] // 2
        input_ids = input_ids[:max_batch_size, :sequence_length]

        attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)

        # generate max 3 tokens
        max_length = 3

        if config.eos_token_id or config.pad_token_id:
            config["pad_token_id"] = config["eos_token_id"]

        return config, input_ids, attention_mask, max_length

    @slow
    def test_model_from_pretrained(self):
        for model_name in OPT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = OPTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class OPTCompatibilityTest(unittest.TestCase):
    test_model_id = "hf-internal-testing/tiny-random-OPTModel"

    @require_package("transformers", "torch")
    def test_model_config_mapping(self):
        # 1. create commmon input
        input_ids = np.random.randint(100, 200, [1, 20])

        # 2. forward the torch model
        import torch
        import transformers

        torch_model_class = getattr(transformers, "OPTModel")
        torch_model = torch_model_class.from_pretrained(OPTCompatibilityTest.test_model_id)
        torch_model.eval()

        torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

        # 3. forward the paddle model
        from paddlenlp import transformers

        paddle_model_class = getattr(transformers, "OPTModel")
        paddle_model = paddle_model_class.from_pretrained(OPTCompatibilityTest.test_model_id, from_hf_hub=True)
        paddle_model.eval()

        paddle_logit = paddle_model(paddle.to_tensor(input_ids), return_dict=False)[0]

        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                atol=1e-4,
            )
        )


class OPTModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_attention(self):
        model = OPTModel.from_pretrained("facebook/opt-1.3b")
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)

        expected_shape = [1, 11, 2048]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [0.81907797, -1.08688772, 1.26071370],
                    [0.96454084, -0.42267877, 1.70609033],
                    [0.78616256, -0.27438506, 0.74083930],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

    @slow
    def test_inference_with_attention(self):
        model = OPTModel.from_pretrained("facebook/opt-1.3b")
        model.eval()

        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
        expected_shape = [1, 11, 2048]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [0.15988758, -0.21016182, -0.28532112],
                    [-0.18293847, -0.35511413, 0.56858277],
                    [0.39969346, -0.33906624, -0.43125907],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


# class OPTGenerationD2STest(GenerationD2STestMixin, unittest.TestCase):
#    internal_testing_model = "__internal_testing__/tiny-random-opt"
#    TokenizerClass = GPTTokenizer
