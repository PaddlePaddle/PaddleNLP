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

import datetime
import math
import unittest
import numpy as np
import random

from tests.testing_utils import slow

from ..test_generation_utils import GenerationTesterMixin
from ..test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask

import paddle
from paddlenlp.transformers import (
    GPTForSequenceClassification,
    GPTForTokenClassification,
    GPTLMHeadModel,
    GPTModel,
    GPTTokenizer,
)

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2-small-en",
    "gpt2-en",
    "gpt2-medium-en",
    "gpt2-large-en",
    "gpt2-xl-en",
    "gpt3-1.3B-en",
    "gpt3-13B-en",
    "gpt-cpm-small-cn-distill",
    "gpt-cpm-large-cn",
]


class GPTModelTester:

    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
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
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = None
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length],
                               self.vocab_size,
                               dtype="int64")

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask(
                [self.batch_size, self.seq_length], dtype="int64")

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
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor(
            [self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = paddle.cast(ids_tensor(
            [self.batch_size, self.seq_length], vocab_size=2),
                                             dtype="float32")

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

    def create_and_check_gpt_model(self, config, input_ids, input_mask, *args):
        model = GPTModel(**config)
        model.eval()

        result = model(input_ids, use_cache=True)
        result = model(input_ids, use_cache=True)
        result = model(input_ids, use_cache=True)

        self.parent.assertEqual(
            result[0].shape,
            [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(len(result[1]), config["num_hidden_layers"])

    def create_and_check_gpt_model_past(self, config, input_ids, input_mask,
                                        *args):
        model = GPTModel(**config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids, use_cache=True)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))

        output, past = outputs

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1),
                                 config["vocab_size"],
                                 dtype="int64")
        next_token_types = ids_tensor([self.batch_size, 1],
                                      self.type_vocab_size,
                                      dtype="int64")

        # append to next input_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)

        output_from_no_past = model(next_input_ids)
        output_from_past = model(next_tokens, use_cache=True, cache=past)[0]

        # select random slice
        random_slice_idx = ids_tensor((1, ), output_from_past.shape[-1]).item()
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

    def create_and_check_gpt_model_attention_mask_past(self, config, input_ids,
                                                       input_mask, *args):
        model = GPTModel(**config)
        model.eval()

        # create attention mask
        attn_mask = paddle.ones(input_ids.shape, dtype="float32")
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
             paddle.ones((attn_mask.shape[0], 1), dtype="float32")],
            axis=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)
        output_from_past = model(next_tokens,
                                 cache=past,
                                 use_cache=True,
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

    def create_and_check_gpt_model_past_large_inputs(self, config, input_ids,
                                                     input_mask, *args):
        model = GPTModel(**config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True)

        output, past = outputs

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3),
                                 config["vocab_size"],
                                 dtype="int64")
        next_token_types = ids_tensor([self.batch_size, 3],
                                      self.type_vocab_size,
                                      dtype="int64")
        next_mask = ids_tensor((self.batch_size, 3),
                               vocab_size=2,
                               dtype="int64")

        # append to next input_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], axis=-1)

        output_from_no_past = model(next_input_ids,
                                    attention_mask=next_attention_mask)
        output_from_past = model(next_tokens,
                                 attention_mask=next_attention_mask,
                                 cache=past,
                                 use_cache=True)[0]
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
        base_model = GPTModel(**config)
        model = GPTLMHeadModel(base_model)
        model.eval()

        result = model(input_ids, use_cache=True)[0]
        self.parent.assertEqual(
            result.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_forward_and_backwards(self, config, input_ids,
                                               input_mask, *args):
        base_model = GPTModel(**config)
        model = GPTLMHeadModel(base_model)

        loss_fct = paddle.nn.loss.CrossEntropyLoss()

        logits = model(input_ids)
        loss = loss_fct(logits, input_ids)
        self.parent.assertEqual(loss.shape, [1])
        self.parent.assertEqual(
            logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        loss.backward()

    def create_and_check_gpt_for_sequence_classification(
            self, config, input_ids, input_mask, sequence_labels, *args):
        base_model = GPTModel(**config)
        model = GPTForSequenceClassification(base_model, self.num_labels)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.shape,
                                [self.batch_size, self.num_labels])

    def create_and_check_gpt_for_token_classification(self, config, input_ids,
                                                      input_mask,
                                                      sequence_labels, *args):
        # config.num_labels = self.num_labels
        base_model = GPTModel(**config)
        model = GPTForTokenClassification(base_model, self.num_labels)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(
            result.shape, [self.batch_size, self.seq_length, self.num_labels])

    def create_and_check_gpt_weight_initialization(self, config, *args):
        model = GPTModel(**config)
        model_std = model.config["initializer_range"] / math.sqrt(
            2 * model.config["num_hidden_layers"])
        for key in model.state_dict().keys():
            if "out_proj" in key and "weight" in key:
                self.parent.assertLessEqual(
                    abs((paddle.std(model.state_dict()[key]) -
                         model_std).numpy()), 0.02)
                self.parent.assertLessEqual(
                    abs((paddle.mean(model.state_dict()[key]) - 0.0).numpy()),
                    0.01)

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


class GPTModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = GPTModel

    all_model_classes = (GPTModel, GPTLMHeadModel, GPTForSequenceClassification,
                         GPTForTokenClassification)
    all_generative_model_classes = {GPTLMHeadModel: (GPTModel, "gpt")}
    all_parallelizable_model_classes = (GPTLMHeadModel)
    test_missing_keys = False
    test_model_parallel = True

    # special case for DoubleHeads model
    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class)

        return inputs_dict

    def setUp(self):
        self.model_tester = GPTModelTester(self)
        random.seed(128)
        np.random.seed(128)
        paddle.seed(128)

    def test_gpt_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_model(*config_and_inputs)

    def test_gpt_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_model_past(*config_and_inputs)

    def test_gpt_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_model_attention_mask_past(
            *config_and_inputs)

    def test_gpt_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_model_past_large_inputs(
            *config_and_inputs)

    def test_gpt_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_gpt_sequence_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_for_sequence_classification(
            *config_and_inputs)

    def test_gpt_token_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_for_token_classification(
            *config_and_inputs)

    def test_gpt_weight_initialization(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_weight_initialization(
            *config_and_inputs)

    @slow
    def test_batch_generation(self):
        model = GPTLMHeadModel.from_pretrained("gpt2-en")
        model.eval()
        tokenizer = GPTTokenizer.from_pretrained("gpt2-en")

        tokenizer.padding_side = "left"

        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.pad_token_id = model.eos_token_id
        getattr(model,
                model.base_model_prefix).config["pad_token_id"] = getattr(
                    model, model.base_model_prefix).config["eos_token_id"]

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I",
        ]

        inputs = tokenizer(sentences,
                           return_tensors="pd",
                           padding=True,
                           return_attention_mask=True,
                           return_position_ids=True)
        input_ids = inputs["input_ids"]

        outputs, _ = model.generate(
            input_ids=input_ids,
            position_ids=inputs["position_ids"],
            decode_strategy="greedy_search",
            attention_mask=inputs["attention_mask"],
            use_cache=True,
        )
        batch_out_sentence = tokenizer.batch_decode(outputs,
                                                    skip_special_tokens=True)

        inputs_non_padded = tokenizer(sentences[0],
                                      return_tensors="pd")["input_ids"]
        output_non_padded, _ = model.generate(input_ids=inputs_non_padded,
                                              use_cache=True,
                                              decode_strategy="greedy_search")
        non_padded_sentence = tokenizer.decode(output_non_padded[0],
                                               skip_special_tokens=True)

        inputs_padded = tokenizer(sentences[1],
                                  return_tensors="pd")["input_ids"]
        output_padded, _ = model.generate(input_ids=inputs_padded,
                                          use_cache=True,
                                          decode_strategy="greedy_search")
        padded_sentence = tokenizer.decode(output_padded[0],
                                           skip_special_tokens=True)

        expected_output_sentence = [
            " bit of a mess. I'm not sure if he's going to be able to walk or not",
            "'m going to be doing a lot of research on this. I'm going to be doing a lot",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(expected_output_sentence,
                             [non_padded_sentence, padded_sentence])

    @slow
    def test_model_from_pretrained(self):
        for model_name in GPT2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = GPTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class GPTModelLanguageGenerationTest(unittest.TestCase):

    def _test_lm_generate_gpt_helper(
        self,
        verify_outputs=True,
    ):
        model = GPTLMHeadModel.from_pretrained("gpt2-en")
        model.eval()

        # The dog
        input_ids = paddle.to_tensor([[464, 3290]], dtype="int64")

        # The dog was found in a field near the intersection of West and West Streets.\n\nThe dog
        # fmt: off
        expected_output_ids = [
            373,
            1043,
            287,
            257,
            2214,
            1474,
            262,
            16246,
            286,
            2688,
            290,
            2688,
            27262,
            13,
            198,
            198,
            464,
            3290,
        ]
        # fmt: on
        output_ids, _ = model.generate(input_ids,
                                       decode_strategy="greedy_search",
                                       max_length=18)
        if verify_outputs:
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @slow
    def test_lm_generate_gpt(self):
        self._test_lm_generate_gpt_helper()

    @slow
    def test_gpt_sample(self):
        tokenizer = GPTTokenizer.from_pretrained("gpt2-en")
        model = GPTLMHeadModel.from_pretrained("gpt2-en")
        model.eval()

        paddle.seed(128)
        np.random.seed(128)
        random.seed(128)

        tokenized = tokenizer("Today is a nice day and",
                              return_tensors="pd",
                              return_position_ids=True,
                              return_attention_mask=True)
        input_ids = tokenized["input_ids"]

        output_ids, _ = model.generate(
            input_ids,
            attention_mask=tokenized["attention_mask"],
            position_ids=tokenized["position_ids"],
            decode_strategy="sampling",
            top_k=1)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        output_seq, _ = model.generate(
            input_ids=input_ids,
            attention_mask=tokenized["attention_mask"],
            position_ids=tokenized["position_ids"],
            decode_strategy="sampling",
            top_k=1,
            num_return_sequences=5)
        output_seq_strs = tokenizer.batch_decode(output_seq,
                                                 skip_special_tokens=True)

        EXPECTED_OUTPUT_STR = (
            " I'm glad I'm here. I'm glad I'm here. I'm glad I'm here")
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)

    @slow
    def test_gpt_sample_max_time(self):
        # NOTE: duration changed sharply and can not be limit in a range for now.
        tokenizer = GPTTokenizer.from_pretrained("gpt2-en")
        model = GPTLMHeadModel.from_pretrained("gpt2-en")
        model.eval()

        paddle.seed(0)
        np.random.seed(0)
        random.seed(0)

        tokenized = tokenizer("Today is a nice day and", return_tensors="pd")
        input_ids = tokenized["input_ids"]

        MAX_TIME = 0.5

        start = datetime.datetime.now()
        model.generate(input_ids,
                       decode_strategy="sampling",
                       max_time=MAX_TIME,
                       max_length=256)
        duration = datetime.datetime.now() - start
        # self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        # self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))

        start = datetime.datetime.now()
        model.generate(input_ids,
                       decode_strategy="greedy_search",
                       max_time=MAX_TIME,
                       max_length=256)
        duration = datetime.datetime.now() - start
        # self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        # self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))

        start = datetime.datetime.now()
        model.generate(input_ids,
                       decode_strategy="beam_search",
                       num_beams=2,
                       max_time=MAX_TIME,
                       max_length=256)
        duration = datetime.datetime.now() - start
        # self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        # self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))
