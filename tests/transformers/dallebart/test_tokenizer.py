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
import json
import os
import unittest

from paddlenlp.transformers import DalleBartTokenizer

from ..test_tokenizer_common import TokenizerTesterMixin

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "wiki_word_frequency_file": "enwiki-words-frequency.txt",
}


class TestTokenizationDalleBart(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = DalleBartTokenizer
    test_rust_tokenizer = False
    test_offsets = False

    def setUp(self):
        super().setUp()
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
            "<mask>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        frequency = ["l 3123", "o 2133", "w 897", "r 1348", "e 6813", "s 7318", "t 1390"]

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        self.wiki_word_frequency_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["wiki_word_frequency_file"])
        self.special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "cls_token": "<s>",
            "sep_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
        }

        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))
        with open(self.wiki_word_frequency_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(frequency))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return "lower newer", "lower newer"

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                # Test not batched,should be processed before encode
                encoded_sequences_1 = tokenizer.encode(
                    tokenizer.text_processor(sequences[0]),
                    max_length=64,  # default
                    padding="max_length",  # default
                    truncation=True,  # default)
                    return_token_type_ids=False,
                    return_attention_mask=True,
                )
                encoded_sequences_2 = tokenizer(sequences[0], return_token_type_ids=False, return_attention_mask=True)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test not batched pairs
                encoded_sequences_1 = tokenizer.encode(
                    tokenizer.text_processor(sequences[0]),
                    tokenizer.text_processor(sequences[1]),
                    max_length=64,  # default
                    padding="max_length",  # default
                    truncation=True,  # default)
                    return_token_type_ids=False,
                    return_attention_mask=True,
                )
                encoded_sequences_2 = tokenizer(
                    sequences[0], sequences[1], return_token_type_ids=False, return_attention_mask=True
                )
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched
                processed_seq = [tokenizer.text_processor(s) for s in sequences]
                encoded_sequences_1 = tokenizer.batch_encode(
                    processed_seq,
                    max_length=64,  # default
                    padding="max_length",  # default
                    truncation=True,  # default)
                    return_token_type_ids=False,
                    return_attention_mask=True,
                )
                encoded_sequences_2 = tokenizer(sequences, return_token_type_ids=False, return_attention_mask=True)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched pairs
                encoded_sequences_1 = tokenizer.batch_encode(
                    list(zip(processed_seq, processed_seq)),
                    max_length=64,  # default
                    padding="max_length",  # default
                    truncation=True,  # default)
                    return_token_type_ids=False,
                    return_attention_mask=True,
                )
                encoded_sequences_2 = tokenizer(
                    sequences, sequences, return_token_type_ids=False, return_attention_mask=True
                )
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

    def test_consecutive_unk_string(self):
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            tokens = [tokenizer.unk_token for _ in range(2)]
            string = tokenizer.convert_tokens_to_string(tokens)
            encoding = tokenizer.encode(
                text=string, runcation=True, return_offsets_mapping=True, padding=False, truncation=False
            )
            self.assertEqual(len(encoding["input_ids"]), 4)
            self.assertEqual(len(encoding["offset_mapping"]), 2)

    def test_padding_to_multiple_of(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.pad_token is None:
                    self.skipTest("No padding token.")
                else:
                    empty_tokens = tokenizer.encode("", padding=True, pad_to_multiple_of=8)
                    normal_tokens = tokenizer.encode("This is a sample input", padding=True, pad_to_multiple_of=8)
                    for key, value in empty_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    normal_tokens = tokenizer.encode("This", pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertNotEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    # Should also work with truncation
                    normal_tokens = tokenizer.encode("This", padding=True, truncation=True, pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    # truncation to something which is not a multiple of pad_to_multiple_of raises an error
                    self.assertRaises(
                        ValueError,
                        tokenizer.__call__,
                        "This",
                        padding=True,
                        truncation=True,
                        max_length=12,
                        pad_to_multiple_of=8,
                    )

    # __call__(),max_length default 64
    def test_maximum_encoding_length_pair_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Build a sequence from our model's vocabulary
                stride = 2
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)
                if len(ids) <= 2 + stride:
                    seq_0 = (seq_0 + " ") * (2 + stride)
                    ids = None

                seq0_tokens = tokenizer.encode(seq_0, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]
                self.assertGreater(len(seq0_tokens), 2 + stride)

                seq_1 = "This is another sentence to be encoded."
                seq1_tokens = tokenizer.encode(seq_1, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]
                if abs(len(seq0_tokens) - len(seq1_tokens)) <= 2:
                    seq1_tokens = seq1_tokens + seq1_tokens
                    seq_1 = tokenizer.decode(seq1_tokens, clean_up_tokenization_spaces=False)
                seq1_tokens = tokenizer.encode(seq_1, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]

                self.assertGreater(len(seq1_tokens), 2 + stride)

                # We are not using the special tokens - a bit too hard to test all the tokenizers with this
                # TODO try this again later
                sequence = tokenizer.encode(seq_0, seq_1, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]  # , add_prefix_space=False)

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_2 = seq_0 * model_max_length
                self.assertGreater(len(seq_2), model_max_length)

                sequence1 = tokenizer.encode(
                    seq_1, return_token_type_ids=None, add_special_tokens=False, truncation=False
                )
                total_length1 = len(sequence1["input_ids"])
                sequence2 = tokenizer.encode(
                    seq_2, seq_1, return_token_type_ids=None, add_special_tokens=False, truncation=False
                )
                total_length2 = len(sequence2["input_ids"])
                self.assertLess(
                    total_length1, model_max_length - 10, "Issue with the testing sequence, please update it."
                )
                self.assertGreater(
                    total_length2, model_max_length, "Issue with the testing sequence, please update it."
                )

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token and tokenizer.pad_token_id >= 0 else [False]
                )
                for padding_state in padding_strategies:
                    with self.subTest(f"{tokenizer.__class__.__name__} Padding: {padding_state}"):
                        for truncation_state in [True, "longest_first", "only_first"]:
                            with self.subTest(f"{tokenizer.__class__.__name__} Truncation: {truncation_state}"):

                                output = tokenizer.encode(
                                    seq_2,
                                    seq_1,
                                    padding=padding_state,
                                    truncation=truncation_state,
                                    max_length=model_max_length,
                                )

                                self.assertEqual(len(output["input_ids"]), model_max_length)
                                output = tokenizer(
                                    [seq_2],
                                    [seq_1],
                                    padding=padding_state,
                                    truncation=truncation_state,
                                    max_length=model_max_length,
                                )
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple
                        output = tokenizer.encode(
                            seq_1, seq_2, padding=padding_state, truncation="only_second", max_length=model_max_length
                        )
                        self.assertEqual(len(output["input_ids"]), model_max_length)

                        output = tokenizer(
                            [seq_1],
                            [seq_2],
                            padding=padding_state,
                            truncation="only_second",
                            max_length=model_max_length,
                        )
                        self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("PaddleNLP", level="WARNING") as cm:
                            output = tokenizer.encode(seq_1, seq_2, padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("PaddleNLP", level="WARNING") as cm:
                            output = tokenizer(
                                [seq_1], [seq_2], padding=padding_state, max_length=None, truncation=False
                            )
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                truncated_first_sequence = (
                    tokenizer.encode(seq_0, return_token_type_ids=None, add_special_tokens=False)["input_ids"][:-2]
                    + tokenizer.encode(seq_1, return_token_type_ids=None, add_special_tokens=False)["input_ids"]
                )
                truncated_second_sequence = (
                    tokenizer.encode(seq_0, return_token_type_ids=None, add_special_tokens=False)["input_ids"]
                    + tokenizer.encode(seq_1, return_token_type_ids=None, add_special_tokens=False)["input_ids"][:-2]
                )

                # TODO(wj-Mcat): `overflow_first_sequence` and `overflow_second_sequence` is not used
                # to make CI green, the following codes will be commented out

                # overflow_first_sequence = (
                #     tokenizer.encode(seq_0, return_token_type_ids=None, add_special_tokens=False)["input_ids"][
                #         -(2 + stride) :
                #     ]
                #     + tokenizer.encode(seq_1, return_token_type_ids=None, add_special_tokens=False)["input_ids"]
                # )
                # overflow_second_sequence = (
                #     tokenizer.encode(seq_0, return_token_type_ids=None, add_special_tokens=False)["input_ids"]
                #     + tokenizer.encode(seq_1, return_token_type_ids=None, add_special_tokens=False)["input_ids"][
                #         -(2 + stride) :
                #     ]
                # )

                with self.assertRaises(ValueError) as context:
                    tokenizer.encode(
                        seq_0,
                        seq_1,
                        max_length=len(sequence) - 2,
                        return_token_type_ids=None,
                        add_special_tokens=False,
                        stride=stride,
                        truncation="longest_first",
                        return_overflowing_tokens=True,
                        # add_prefix_space=False,
                    )

                self.assertTrue(
                    context.exception.args[0].startswith(
                        "Not possible to return overflowing tokens for pair of sequences with the "
                        "`longest_first`. Please select another truncation strategy than `longest_first`, "
                        "for instance `only_second` or `only_first`."
                    )
                )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                # No overflowing tokens when using 'longest' in python tokenizers
                with self.assertRaises(ValueError) as context:
                    tokenizer.encode(
                        seq_0,
                        seq_1,
                        max_length=len(sequence) - 2,
                        return_token_type_ids=None,
                        add_special_tokens=False,
                        stride=stride,
                        truncation=True,
                        return_overflowing_tokens=True,
                        # add_prefix_space=False,
                    )

                self.assertTrue(
                    context.exception.args[0].startswith(
                        "Not possible to return overflowing tokens for pair of sequences with the "
                        "`longest_first`. Please select another truncation strategy than `longest_first`, "
                        "for instance `only_second` or `only_first`."
                    )
                )

                information_first_truncated = tokenizer.encode(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                truncated_sequence = information_first_truncated["input_ids"]
                overflowing_tokens = information_first_truncated["overflowing_tokens"]

                self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                self.assertEqual(truncated_sequence, truncated_first_sequence)

                self.assertEqual(len(overflowing_tokens), 2 + stride)
                self.assertEqual(overflowing_tokens, seq0_tokens[-(2 + stride) :])

                information_second_truncated = tokenizer.encode(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_second",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                truncated_sequence = information_second_truncated["input_ids"]
                overflowing_tokens = information_second_truncated["overflowing_tokens"]

                self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                self.assertEqual(truncated_sequence, truncated_second_sequence)

                self.assertEqual(len(overflowing_tokens), 2 + stride)
                self.assertEqual(overflowing_tokens, seq1_tokens[-(2 + stride) :])

    # __call__(),max_length default 64
    def test_maximum_encoding_length_single_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)

                sequence = tokenizer.encode(seq_0, return_token_type_ids=None, add_special_tokens=False)["input_ids"]
                total_length = len(sequence)

                self.assertGreater(total_length, 4, "Issue with the testing sequence, please update it it's too short")

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_1 = seq_0 * model_max_length

                sequence1 = tokenizer.encode(
                    seq_1, return_token_type_ids=None, add_special_tokens=False, truncation=False
                )
                total_length1 = len(sequence1["input_ids"])
                self.assertGreater(
                    total_length1, model_max_length, "Issue with the testing sequence, please update it it's too short"
                )

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token and tokenizer.pad_token_id >= 0 else [False]
                )
                for padding_state in padding_strategies:
                    with self.subTest(f"Padding: {padding_state}"):
                        for truncation_state in [True, "longest_first", "only_first"]:
                            with self.subTest(f"Truncation: {truncation_state}"):
                                output = tokenizer.encode(seq_1, padding=padding_state, truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"]), model_max_length)

                                output = tokenizer(
                                    [seq_1],
                                    padding=padding_state,
                                    max_length=model_max_length,
                                    truncation=truncation_state,
                                )
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("PaddleNLP", level="WARNING") as cm:
                            output = tokenizer.encode(seq_1, padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("PaddleNLP", level="WARNING") as cm:
                            output = tokenizer([seq_1], padding=padding_state, truncation=False, max_length=None)
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                # Overflowing tokens
                stride = 2
                information = tokenizer.encode(
                    seq_0,
                    max_length=total_length - 2,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="longest_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                truncated_sequence = information["input_ids"]
                overflowing_tokens = information["overflowing_tokens"]

                self.assertEqual(len(truncated_sequence), total_length - 2)
                self.assertEqual(truncated_sequence, sequence[:-2])

                self.assertEqual(len(overflowing_tokens), 2 + stride)
                self.assertEqual(overflowing_tokens, sequence[-(2 + stride) :])
