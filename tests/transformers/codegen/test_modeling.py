# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import datetime
import unittest
import numpy as np
import random

import paddle
from paddlenlp.transformers import (CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST,
                                    AutoTokenizer, CodeGenForCausalLM,
                                    CodeGenModel, CodeGenTokenizer)
from ...testing_utils import slow

from ..test_generation_utils import GenerationTesterMixin
from ..test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


class CodeGenModelTester:

    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        use_mc_token_ids=True,
        vocab_size=256,
        hidden_size=32,
        rotary_dim=4,
        num_hidden_layers=5,
        num_attention_heads=4,
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
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.use_mc_token_ids = use_mc_token_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rotary_dim = rotary_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
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

        paddle.seed(128)
        np.random.seed(128)
        random.seed(128)

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length],
                               self.vocab_size,
                               dtype="int64")

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask(
                [self.batch_size, self.seq_length], dtype="int64")

        mc_token_ids = None
        if self.use_mc_token_ids:
            mc_token_ids = ids_tensor([self.batch_size, self.num_choices],
                                      self.seq_length,
                                      dtype="int64")

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size],
                                         self.type_sequence_label_size,
                                         dtype="int64")
            token_labels = ids_tensor([self.batch_size, self.seq_length],
                                      self.num_labels,
                                      dtype="int64")
            choice_labels = ids_tensor([self.batch_size],
                                       self.num_choices,
                                       dtype="int64")

        config = self.get_config()

        return (
            config,
            input_ids,
            input_mask,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "n_embd": self.hidden_size,
            "n_layer": self.num_hidden_layers,
            "n_head": self.num_attention_heads,
            "activation_function": self.hidden_act,
            "resid_pdrop": self.hidden_dropout_prob,
            "attn_pdrop": self.attention_probs_dropout_prob,
            "n_positions": self.max_position_embeddings,
            "n_ctx": self.max_position_embeddings,
            "initializer_range": self.initializer_range,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "rotary_dim": self.rotary_dim,
        }

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor(
            [self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length],
                                            vocab_size=2,
                                            dtype="int64")

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

    def create_and_check_codegen_model(self, config, input_ids, input_mask,
                                       *args):
        model = CodeGenModel(**config)
        model.eval()

        result = model(input_ids, use_cache=True)

        self.parent.assertEqual(
            result[0].shape,
            [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(len(result[1]), config["n_layer"])

    def create_and_check_codegen_model_past(self, config, input_ids, input_mask,
                                            *args):
        model = CodeGenModel(**config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids, )
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past))

        output, past = outputs

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1),
                                 config["vocab_size"],
                                 dtype="int64")

        # append to next input_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids)[0]
        output_from_past = model(next_tokens, cache=past)[0]

        # select random slice
        random_slice_idx = ids_tensor((1, ),
                                      output_from_past.shape[-1],
                                      dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, -1,
                                                        random_slice_idx].detach(
                                                        )
        output_from_past_slice = output_from_past[:, 0,
                                                  random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(
            paddle.allclose(output_from_past_slice,
                            output_from_no_past_slice,
                            atol=1e-3))

    def create_and_check_codegen_model_attention_mask_past(
            self, config, input_ids, input_mask, *args):
        model = CodeGenModel(**config)
        model.eval()

        # create attention mask
        attn_mask = paddle.ones(input_ids.shape, dtype="int64")
        half_seq_length = self.seq_length // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past = model(input_ids,
                             attention_mask=attn_mask,
                             use_cache=True)

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1),
                                 config["vocab_size"],
                                 dtype="int64")

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor(
            (1, ), half_seq_length, dtype="int64").item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1),
                                              config["vocab_size"],
                                              dtype="int64").squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        attn_mask = paddle.concat(
            [attn_mask,
             paddle.ones((attn_mask.shape[0], 1), dtype="int64")],
            axis=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)[0]
        output_from_past = model(next_tokens,
                                 cache=past,
                                 attention_mask=attn_mask)[0]

        # select random slice
        random_slice_idx = ids_tensor((1, ),
                                      output_from_past.shape[-1],
                                      dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, -1,
                                                        random_slice_idx].detach(
                                                        )
        output_from_past_slice = output_from_past[:, 0,
                                                  random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(
            paddle.allclose(output_from_past_slice,
                            output_from_no_past_slice,
                            atol=1e-3))

    def create_and_check_codegen_model_past_large_inputs(
            self, config, input_ids, input_mask, *args):
        model = CodeGenModel(**config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True)

        output, past = outputs

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3),
                                 config["vocab_size"],
                                 dtype="int64")
        next_mask = ids_tensor((self.batch_size, 3),
                               vocab_size=2,
                               dtype="int64")

        # append to next input_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], axis=-1)

        output_from_no_past = model(next_input_ids,
                                    attention_mask=next_attention_mask)[0]
        output_from_past = model(next_tokens,
                                 attention_mask=next_attention_mask,
                                 cache=past)[0]
        self.parent.assertTrue(
            output_from_past.shape[1] == next_tokens.shape[1])

        # select random slice
        random_slice_idx = ids_tensor((1, ),
                                      output_from_past.shape[-1],
                                      dtype="int64").item()
        output_from_no_past_slice = output_from_no_past[:, -3:,
                                                        random_slice_idx].detach(
                                                        )
        output_from_past_slice = output_from_past[:, :,
                                                  random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(
            paddle.allclose(output_from_past_slice,
                            output_from_no_past_slice,
                            atol=1e-3))

    def create_and_check_lm_head_model(self, config, input_ids, input_mask,
                                       *args):
        base_model = CodeGenModel(**config)
        model = CodeGenForCausalLM(base_model)

        loss_fct = paddle.nn.CrossEntropyLoss()

        logits, cache = model(input_ids)
        loss = loss_fct(logits[:, :-1, :], input_ids[:, 1:])
        self.parent.assertEqual(loss.shape, [1])
        self.parent.assertEqual(
            logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_forward_and_backwards(self, config, input_ids,
                                               input_mask, *args):
        base_model = CodeGenModel(**config)
        model = CodeGenForCausalLM(base_model)

        loss_fct = paddle.nn.CrossEntropyLoss()
        logits, cache = model(input_ids)
        loss = loss_fct(logits[:, :-1, :], input_ids[:, 1:])
        self.parent.assertEqual(loss.shape, [1])
        self.parent.assertEqual(
            logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        result.loss.backward()

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        inputs_dict = {"input_ids": input_ids}

        return config, inputs_dict


class CodeGenModelTest(ModelTesterMixin, GenerationTesterMixin,
                       unittest.TestCase):
    base_model_class = CodeGenModel

    all_model_classes = (CodeGenModel, CodeGenForCausalLM)
    all_generative_model_classes = {
        CodeGenForCausalLM: (CodeGenModel, "transformer")
    }
    fx_compatible = False
    test_pruning = False
    test_missing_keys = False
    test_model_parallel = False
    test_head_masking = False

    # attention mask issue
    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(
        )

        input_ids = inputs_dict[self.input_name]
        attention_mask = paddle.zeros_like(input_ids, dtype=paddle.float32)

        max_batch_size = 2
        sequence_length = input_ids.shape[-1] // 2
        input_ids = input_ids[:max_batch_size, :sequence_length]
        attention_mask = attention_mask[:max_batch_size, :
                                        sequence_length].unsqueeze([1, 2])

        # generate max 3 tokens
        max_length = 3

        if config.get(
                "eos_token_id",
                None) is not None and config.get("pad_token_id", None) is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            config["pad_token_id"] = config["eos_token_id"]

        return config, input_ids, attention_mask, max_length

    # special case for DoubleHeads model
    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class)
        return inputs_dict

    def setUp(self):
        self.model_tester = CodeGenModelTester(self)

    def test_codegen_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_codegen_model(*config_and_inputs)

    def test_codegen_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_codegen_model_past(
            *config_and_inputs)

    def test_codegen_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_codegen_model_attention_mask_past(
            *config_and_inputs)

    def test_codegen_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_codegen_model_past_large_inputs(
            *config_and_inputs)

    def test_codegen_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    @slow
    def test_batch_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/codegen-350M-mono")
        model = CodeGenForCausalLM.from_pretrained(
            "Salesforce/codegen-350M-mono")
        model.eval()

        tokenizer.padding_side = "left"

        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.transformer.config["pad_token_id"] = model.transformer.config[
            "eos_token_id"]

        # use different length sentences to test batching
        sentences = ["def hellow_world():", "def greet(name):"]

        inputs = tokenizer(sentences,
                           return_tensors="pd",
                           padding=True,
                           return_attention_mask=True)
        input_ids = inputs["input_ids"]

        outputs, _ = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
        )

        inputs_non_padded = tokenizer(sentences[0],
                                      return_tensors="pd")["input_ids"]
        output_non_padded, _ = model.generate(input_ids=inputs_non_padded)

        inputs_padded = tokenizer(sentences[1],
                                  return_tensors="pd")["input_ids"]
        output_padded, _ = model.generate(input_ids=inputs_padded)

        batch_out_sentence = tokenizer.batch_decode(outputs,
                                                    skip_special_tokens=True)

        non_padded_sentence = tokenizer.decode(output_non_padded[0],
                                               skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0],
                                           skip_special_tokens=True)

        expected_output_sentence = [
            '\n      print("Hello World")\n\nhellow_world()\n\n#',
            '\n      print(f"Hello {name}")\n\ngreet("Rolf")\n',
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)

        self.assertListEqual(expected_output_sentence,
                             [non_padded_sentence, padded_sentence])

    @slow
    def test_model_from_pretrained(self):
        for model_name in CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CodeGenModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_model_name_list(self):
        pass

    @slow
    def test_auto_tokenizer(self):
        for model_name in CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST:
            tokenizer = AutoTokenizer.from_pretrained(model_name)


class CodeGenModelLanguageGenerationTest(unittest.TestCase):

    @slow
    def test_lm_generate_codegen(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/codegen-350M-mono")
        model = CodeGenForCausalLM.from_pretrained(
            "Salesforce/codegen-350M-mono")
        model.eval()

        inputs = tokenizer("def hello_world():",
                           return_tensors="pd",
                           return_attention_mask=True,
                           return_token_type_ids=False)
        expected_output = '\n      print("Hello World")\n\nhello_world()\n\n#'

        output_ids, _ = model.generate(**inputs,
                                       decode_strategy="sampling",
                                       top_k=1)
        output_str = tokenizer.batch_decode(output_ids)[0]

        self.assertEqual(output_str, expected_output)

    @slow
    def test_codegen_sample(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/codegen-350M-mono")
        model = CodeGenForCausalLM.from_pretrained(
            "Salesforce/codegen-350M-mono")
        model.eval()

        tokenized = tokenizer("def hello_world():",
                              return_tensors="pd",
                              return_token_type_ids=True,
                              return_attention_mask=True)
        input_ids = tokenized["input_ids"]
        output_ids, _ = model.generate(input_ids,
                                       decode_strategy="sampling",
                                       top_k=1)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        token_type_ids = tokenized.token_type_ids
        output_seq, _ = model.generate(input_ids=input_ids,
                                       decode_strategy="sampling",
                                       top_k=1,
                                       num_return_sequences=5)
        output_seq_tt, _ = model.generate(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          decode_strategy="sampling",
                                          top_k=1,
                                          num_return_sequences=5)
        output_seq_strs = tokenizer.batch_decode(output_seq,
                                                 skip_special_tokens=True)
        output_seq_tt_strs = tokenizer.batch_decode(output_seq_tt,
                                                    skip_special_tokens=True)

        EXPECTED_OUTPUT_STR = '\n      print("Hello World")\n\nhello_world()\n\n#'

        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)
        self.assertTrue(
            all([
                output_seq_strs[idx] != output_seq_tt_strs[idx]
                for idx in range(len(output_seq_tt_strs))
            ]))  # token_type_ids should change output
