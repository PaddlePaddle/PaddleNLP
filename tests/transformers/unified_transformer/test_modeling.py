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

import datetime
import math
import unittest
import numpy as np
import random

from tests.testing_utils import slow
from parameterized import parameterized_class

from ..test_generation_utils import GenerationTesterMixin
from ..test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask

import paddle
import paddle.nn as nn
from paddlenlp.transformers import (
    UnifiedTransformerModel,
    UnifiedTransformerLMHeadModel,
    UnifiedTransformerForMaskedLM,
    UnifiedTransformerTokenizer,
)
from paddlenlp.data import Pad
from paddlenlp.data import DataCollatorWithPadding

UNIFIED_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "unified_transformer-12L-cn",
    "unified_transformer-12L-cn-luge",
    "plato-mini",
]


def batchify_fn(batch_examples, pad_val):

    def pad_mask(batch_attention_mask):
        batch_size = len(batch_attention_mask)
        max_len = max(map(len, batch_attention_mask))
        attention_mask = np.ones(
            (batch_size, max_len, max_len), dtype='float32') * -1e4
        for i, mask_data in enumerate(attention_mask):
            seq_len = len(batch_attention_mask[i])
            mask_data[-seq_len:, -seq_len:] = np.array(batch_attention_mask[i],
                                                       dtype='float32')
        # In order to ensure the correct broadcasting mechanism, expand one
        # dimension to the second dimension (n_head of Transformer).
        attention_mask = np.expand_dims(attention_mask, axis=1)
        return attention_mask

    pad_func = Pad(pad_val=pad_val, pad_right=False, dtype='int64')

    input_ids = pad_func([example['input_ids'] for example in batch_examples])
    token_type_ids = pad_func(
        [example['token_type_ids'] for example in batch_examples])
    position_ids = pad_func(
        [example['position_ids'] for example in batch_examples])

    attention_mask = pad_mask(
        [example['attention_mask'] for example in batch_examples])

    return {
        "input_ids": paddle.to_tensor(input_ids, dtype="int64"),
        "token_type_ids": paddle.to_tensor(token_type_ids, dtype="int64"),
        "position_ids": paddle.to_tensor(position_ids, dtype="int64"),
        "attention_mask": paddle.to_tensor(attention_mask, dtype="float32")
    }


def postprocess_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.sep_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    response = " ".join(tokens)
    return response


class UnifiedTransformerModelTester:

    def __init__(self,
                 parent,
                 is_training=True,
                 batch_size=14,
                 seq_length=7,
                 vocab_size=99,
                 hidden_size=32,
                 num_hidden_layers=5,
                 num_attention_heads=4,
                 intermediate_size=37,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 normalize_before=True,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 unk_token_id=0,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 role_type_size=None):
        self.parent = parent
        self.is_training = is_training
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.normalize_before = normalize_before
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = vocab_size - 1
        self.role_type_size = role_type_size

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length],
                               self.vocab_size,
                               dtype="int64")
        input_mask = random_attention_mask([self.batch_size, self.seq_length],
                                           dtype="int64").unsqueeze([1, 2])
        token_type_ids = ids_tensor([self.batch_size, self.seq_length],
                                    self.type_vocab_size,
                                    dtype="int64")
        position_ids = paddle.tile(
            paddle.arange(end=self.seq_length, dtype="int64").reshape([1, -1]),
            [self.batch_size, 1])

        lm_labels = None
        if self.parent.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.seq_length],
                                   self.vocab_size)

        config = self.get_config()

        return (config, input_ids, input_mask, token_type_ids, position_ids,
                lm_labels)

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
            "normalize_before": self.normalize_before,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "unk_token_id": self.unk_token_id,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "mask_token_id": self.mask_token_id,
            "role_type_size": self.role_type_size
        }

    def prepare_config_and_inputs_for_decoder(self):
        (config, input_ids, input_mask, token_type_ids, position_ids,
         lm_labels) = self.prepare_config_and_inputs()
        return (config, input_ids, input_mask, token_type_ids, position_ids,
                lm_labels)

    def create_and_check_unified_transformer_model(self, config, input_ids,
                                                   input_mask, token_type_ids,
                                                   position_ids, *args):
        model = UnifiedTransformerModel(**config)
        model.eval()

        result, cache = model(input_ids,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              attention_mask=input_mask,
                              use_cache=True,
                              return_dict=self.parent.return_dict)[:2]

        self.parent.assertEqual(
            result.shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(len(cache), config["num_hidden_layers"])

    def create_and_check_unified_transformer_model_past(self, config, input_ids,
                                                        input_mask,
                                                        token_type_ids,
                                                        position_ids, *args):
        model = UnifiedTransformerModel(**config)
        model.eval()

        # first forward pass
        outputs = model(input_ids,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        attention_mask=input_mask,
                        use_cache=True,
                        return_dict=self.parent.return_dict)
        outputs_use_cache_conf = model(input_ids,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       attention_mask=input_mask,
                                       return_dict=self.parent.return_dict)
        outputs_no_past = model(input_ids,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                attention_mask=input_mask,
                                use_cache=False,
                                return_dict=self.parent.return_dict)

        self.parent.assertTrue(
            len(outputs_no_past) == len(outputs_use_cache_conf))

        output, past = outputs[:2]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1),
                                 config["vocab_size"],
                                 dtype="int64")
        next_token_types = ids_tensor([self.batch_size, 1],
                                      self.type_vocab_size,
                                      dtype="int64")
        next_position = position_ids[:, -1:] + 1

        # append to next input_ids and token_type_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_token_type_ids = paddle.concat([token_type_ids, next_token_types],
                                            axis=-1)
        next_position_ids = paddle.concat([position_ids, next_position],
                                          axis=-1)

        input_mask_t = paddle.transpose(input_mask, perm=[0, 1, 3, 2])
        input_mask = input_mask * input_mask_t

        next_attention_mask = nn.Pad2D([0, 0, 0, 1],
                                       mode='replicate')(input_mask)
        next_attention_mask = nn.Pad2D([0, 1, 0, 0],
                                       value=0)(next_attention_mask)
        next_attention_mask[:, :, -1, -1] = 1

        output_from_no_past, cache = model(
            next_input_ids,
            token_type_ids=next_token_type_ids,
            position_ids=next_position_ids,
            attention_mask=next_attention_mask,
            use_cache=True,
            return_dict=self.parent.return_dict)[:2]
        output_from_past = model(next_tokens,
                                 token_type_ids=next_token_types,
                                 position_ids=next_position,
                                 attention_mask=next_attention_mask[:, :,
                                                                    -1:, :],
                                 use_cache=True,
                                 cache=past,
                                 return_dict=self.parent.return_dict)[0]

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

    def create_and_check_unified_transformer_model_past_large_inputs(
            self, config, input_ids, input_mask, token_type_ids, position_ids,
            *args):
        model = UnifiedTransformerModel(**config)
        model.eval()

        # first forward pass
        output, past = model(input_ids,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             attention_mask=input_mask,
                             use_cache=True,
                             return_dict=self.parent.return_dict)[:2]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3),
                                 config["vocab_size"],
                                 dtype="int64")
        next_token_types = ids_tensor([self.batch_size, 3],
                                      self.type_vocab_size,
                                      dtype="int64")
        next_position = position_ids[:, -3:] + 3

        # append to next input_ids and token_type_ids
        next_input_ids = paddle.concat([input_ids, next_tokens], axis=-1)
        next_token_type_ids = paddle.concat([token_type_ids, next_token_types],
                                            axis=-1)
        next_position_ids = paddle.concat([position_ids, next_position],
                                          axis=-1)

        input_mask_t = paddle.transpose(input_mask, perm=[0, 1, 3, 2])
        input_mask = input_mask * input_mask_t

        next_attention_mask = nn.Pad2D([0, 0, 0, 3],
                                       mode='replicate')(input_mask)
        next_attention_mask = nn.Pad2D([0, 3, 0, 0],
                                       value=0)(next_attention_mask)
        next_attention_mask[:, :, -1, -1] = 1
        next_attention_mask[:, :, -2, -2] = 1
        next_attention_mask[:, :, -3, -3] = 1
        next_attention_mask[:, :, -2, -1] = 1
        next_attention_mask[:, :, -3, -1] = 1
        next_attention_mask[:, :, -3, -2] = 1

        output_from_no_past = model(next_input_ids,
                                    token_type_ids=next_token_type_ids,
                                    attention_mask=next_attention_mask,
                                    position_ids=next_position_ids,
                                    use_cache=False,
                                    return_dict=self.parent.return_dict)
        if self.parent.return_dict:
            output_from_no_past = output_from_no_past[0]
        output_from_past = model(next_tokens,
                                 token_type_ids=next_token_types,
                                 attention_mask=next_attention_mask[:, :,
                                                                    -3:, :],
                                 position_ids=next_position,
                                 cache=past,
                                 use_cache=True,
                                 return_dict=self.parent.return_dict)[0]
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
                                       token_type_ids, position_ids, lm_labels,
                                       *args):
        base_model = UnifiedTransformerModel(**config)
        model = UnifiedTransformerLMHeadModel(base_model)
        model.eval()

        outputs = model(input_ids,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        attention_mask=input_mask,
                        labels=lm_labels,
                        return_dict=self.parent.return_dict)
        if self.parent.use_labels:
            loss, result = outputs[:2]
            self.parent.assertIsInstance(loss.item(), float)
        else:
            result = outputs[0] if self.parent.return_dict else outputs
        self.parent.assertEqual(
            result.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_forward_and_backwards(self, config, input_ids,
                                               input_mask, token_type_ids,
                                               position_ids, lm_head, *args):
        base_model = UnifiedTransformerModel(**config)
        model = UnifiedTransformerLMHeadModel(base_model)

        loss, logits = model(input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=input_mask,
                             position_ids=position_ids,
                             label=input_ids,
                             return_dict=self.parent.return_dict)[:2]
        self.parent.assertIsInstance(loss.item(), float)
        self.parent.assertEqual(
            logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        loss.backward()

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (config, input_ids, input_mask, token_type_ids, position_ids,
         lm_labels) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
            "position_ids": position_ids
        }

        return config, inputs_dict


@parameterized_class(("return_dict", "use_labels"), [
    [False, False],
    [False, True],
    [True, False],
    [True, True],
])
class UnifiedTransformerModelTest(ModelTesterMixin, GenerationTesterMixin,
                                  unittest.TestCase):
    base_model_class = UnifiedTransformerModel

    all_model_classes = (UnifiedTransformerModel, UnifiedTransformerLMHeadModel)
    all_generative_model_classes = {
        UnifiedTransformerLMHeadModel:
        (UnifiedTransformerModel, "unified_transformer")
    }
    test_missing_keys = False

    use_labels = False
    return_dict = False

    # special case for DoubleHeads model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class)
        return inputs_dict

    def setUp(self):
        seed = 1028
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

        self.model_tester = UnifiedTransformerModelTester(self)

    def test_unified_transformer_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_unified_transformer_model(
            *config_and_inputs)

    def test_unified_transformer_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_unified_transformer_model_past(
            *config_and_inputs)

    def test_unified_transformer_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_unified_transformer_model_past_large_inputs(
            *config_and_inputs)

    def test_unified_transformer_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    @slow
    def test_batch_generation(self):
        model = UnifiedTransformerLMHeadModel.from_pretrained("plato-mini")
        tokenizer = UnifiedTransformerTokenizer.from_pretrained("plato-mini")
        model.eval()

        tokenizer.padding_side = "left"

        # use different length sentences to test batching
        sentences = [
            ["你好"],
            ["今天天气不错"],
        ]
        inputs = []
        for seq in sentences:
            inputs.append(
                tokenizer.dialogue_encode(history=seq,
                                          add_start_token_as_response=True))

        data = batchify_fn(inputs, tokenizer.pad_token_id)

        input_ids = data["input_ids"]
        position_ids = data["position_ids"]
        token_type_ids = data["token_type_ids"]
        attention_mask = data["attention_mask"]

        outputs, _ = model.generate(input_ids=input_ids,
                                    position_ids=position_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    decode_strategy="greedy_search")

        data_non_padded = tokenizer.dialogue_encode(
            sentences[0], add_start_token_as_response=True)
        output_non_padded, _ = model.generate(
            input_ids=paddle.to_tensor(data_non_padded["input_ids"],
                                       dtype="int64").reshape([1, -1]),
            position_ids=paddle.to_tensor(data_non_padded["position_ids"],
                                          dtype="int64").reshape([1, -1]),
            token_type_ids=paddle.to_tensor(data_non_padded["token_type_ids"],
                                            dtype="int64").reshape([1, -1]),
            attention_mask=paddle.to_tensor(data_non_padded["attention_mask"],
                                            dtype="float32").unsqueeze([0, 1]),
            decode_strategy="greedy_search")

        data_padded = tokenizer.dialogue_encode(
            sentences[1], add_start_token_as_response=True)
        output_padded, _ = model.generate(
            input_ids=paddle.to_tensor(data_padded["input_ids"],
                                       dtype="int64").reshape([1, -1]),
            position_ids=paddle.to_tensor(data_padded["position_ids"],
                                          dtype="int64").reshape([1, -1]),
            token_type_ids=paddle.to_tensor(data_padded["token_type_ids"],
                                            dtype="int64").reshape([1, -1]),
            attention_mask=paddle.to_tensor(data_padded["attention_mask"],
                                            dtype="float32").unsqueeze([0, 1]),
            decode_strategy="greedy_search")

        batch_out_sentence = []
        for i in range(len(outputs)):
            batch_out_sentence.append(
                postprocess_response(outputs[i].numpy(), tokenizer))
        non_padded_sentence = postprocess_response(output_non_padded[0],
                                                   tokenizer)
        padded_sentence = postprocess_response(output_padded[0], tokenizer)

        expected_output_sentence = [
            "你好 , 你 是 做 什么 工作 的 ?",
            "是 啊 , 我 也 很开心",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(expected_output_sentence,
                             [non_padded_sentence, padded_sentence])


class UnifiedTransformerModelLanguageGenerationTest(unittest.TestCase):

    def _test_lm_generate_unified_transformer_helper(
        self,
        verify_outputs=True,
    ):
        model = UnifiedTransformerLMHeadModel.from_pretrained("plato-mini")
        model.eval()

        input_ids = paddle.to_tensor([[1, 464, 3290, 2, 1]], dtype="int64")
        position_ids = paddle.to_tensor([[0, 1, 2, 3, 4]], dtype="int64")
        token_type_ids = paddle.to_tensor([[0, 0, 0, 0, 1]], dtype="int64")

        expected_output_ids = [
            9,
            113,
            78,
            48,
            3290,
            4,
            16,
            2,
        ]

        output_ids, _ = model.generate(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            decode_strategy="greedy_search",
        )

        if verify_outputs:
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @slow
    def test_lm_generate_unified_transformer(self):
        self._test_lm_generate_unified_transformer_helper()

    @slow
    def test_unified_transformer_sample(self):
        tokenizer = UnifiedTransformerTokenizer.from_pretrained("plato-mini")
        model = UnifiedTransformerLMHeadModel.from_pretrained("plato-mini")
        model.eval()

        sequence = ["今天天气真好！"]

        tokenized = tokenizer.dialogue_encode(history=sequence,
                                              add_start_token_as_response=True)
        output_ids, _ = model.generate(
            paddle.to_tensor(tokenized["input_ids"],
                             dtype="int64").reshape([1, -1]),
            position_ids=paddle.to_tensor(tokenized["position_ids"],
                                          dtype="int64").reshape([1, -1]),
            token_type_ids=paddle.to_tensor(tokenized["token_type_ids"],
                                            dtype="int64").reshape([1, -1]),
            attention_mask=paddle.to_tensor(tokenized["attention_mask"],
                                            dtype="float32").unsqueeze([0, 1]),
            decode_strategy="sampling",
            top_k=1)
        output_str = postprocess_response(output_ids[0].numpy(), tokenizer)

        EXPECTED_OUTPUT_STR = ("你 在 做 什么 呢 ?")
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)

    def test_generate_without_input_ids(self):
        pass
