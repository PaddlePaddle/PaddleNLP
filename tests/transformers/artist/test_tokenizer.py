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

import os
import re
import unittest

from paddlenlp.transformers import ArtistTokenizer

from ...transformers.test_tokenizer_common import TokenizerTesterMixin


class ArtistTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = ArtistTokenizer
    space_between_special_tokens = True
    test_seq2seq = False

    def setUp(self):
        super().setUp()
        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        self.vocab_file = os.path.join(self.tmpdirname, ArtistTokenizer.resource_files_names["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

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

                # Test not batched
                encoded_sequences_1 = tokenizer.encode(
                    sequences[0],
                    return_token_type_ids=False,
                    return_attention_mask=True,
                    max_length=32,
                    padding="max_length",
                    truncation=True,
                )
                encoded_sequences_2 = tokenizer(sequences[0], return_token_type_ids=False, return_attention_mask=True)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test not batched pairs
                encoded_sequences_1 = tokenizer.encode(
                    sequences[0],
                    sequences[1],
                    return_token_type_ids=False,
                    return_attention_mask=True,
                    max_length=32,
                    padding="max_length",
                    truncation=True,
                )
                encoded_sequences_2 = tokenizer(
                    sequences[0], sequences[1], return_token_type_ids=False, return_attention_mask=True
                )
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched
                encoded_sequences_1 = tokenizer.batch_encode(
                    sequences,
                    return_token_type_ids=False,
                    return_attention_mask=True,
                    max_length=32,
                    padding="max_length",
                    truncation=True,
                )
                encoded_sequences_2 = tokenizer(sequences, return_token_type_ids=False, return_attention_mask=True)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched pairs
                encoded_sequences_1 = tokenizer.batch_encode(
                    list(zip(sequences, sequences)),
                    return_token_type_ids=False,
                    return_attention_mask=True,
                    max_length=32,
                    padding="max_length",
                    truncation=True,
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
            encoding = tokenizer(
                text=string,
                runcation=True,
                return_offsets_mapping=True,
            )
            self.assertEqual(len(encoding["input_ids"]), 32)
            self.assertEqual(len(encoding["offset_mapping"]), 34)

    def test_conversion_reversible(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab = tokenizer.get_vocab()
                for word, ind in vocab.items():
                    if word == tokenizer.unk_token:
                        continue
                    self.assertEqual(tokenizer.convert_tokens_to_ids(word), ind + tokenizer.image_vocab_size)
                    self.assertEqual(tokenizer.convert_ids_to_tokens(ind + tokenizer.image_vocab_size), word)

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5):
        toks = [
            (i, tokenizer.decode([i], clean_up_tokenization_spaces=False))
            for i in range(tokenizer.image_vocab_size, tokenizer.image_vocab_size + len(tokenizer))
        ]
        # filter the english only character
        if self.only_english_character:
            toks = list(filter(lambda t: re.match(r"^[ a-zA-Z]+$", t[1]), toks))

        toks = list(
            filter(
                lambda t: [t[0]]
                == tokenizer.encode(t[1], return_token_type_ids=None, add_special_tokens=False)["input_ids"],
                toks,
            )
        )
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        if min_length is not None and len(toks) < min_length and len(toks) > 0:
            while len(toks) < min_length:
                toks = toks + toks
        # toks_str = [t[1] for t in toks]
        toks_ids = [t[0] for t in toks]

        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt, return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        return output_txt, output_ids

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

                sequence1 = tokenizer(seq_1, return_token_type_ids=None, add_special_tokens=False, truncation=False)
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

                                output = tokenizer(
                                    seq_1,
                                    padding=padding_state,
                                    max_length=model_max_length,
                                    truncation=truncation_state,
                                )
                                self.assertEqual(len(output["input_ids"]), model_max_length)

                                output = tokenizer(
                                    [seq_1],
                                    padding=padding_state,
                                    max_length=model_max_length,
                                    truncation=truncation_state,
                                )
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)

                # Overflowing tokens
                stride = 2
                information = tokenizer(
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

    def test_special_tokens_mask(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                # Testing single inputs
                encoded_sequence = tokenizer.encode(sequence_0, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]
                encoded_sequence_dict = tokenizer.encode(
                    sequence_0, add_special_tokens=True, return_special_tokens_mask=True  # , add_prefix_space=False
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                encoded_sequence_w_special = (
                    [tokenizer.cls_token_id] + encoded_sequence_w_special + [tokenizer.cls_token_id]
                )
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))
                filtered_sequence = [x for i, x in enumerate(encoded_sequence_w_special) if not special_tokens_mask[i]]
                self.assertEqual(encoded_sequence, filtered_sequence)

    def test_padding_to_multiple_of(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.pad_token is None:
                    self.skipTest("No padding token.")
                else:
                    empty_tokens = tokenizer("", padding=True, pad_to_multiple_of=8)
                    normal_tokens = tokenizer("This is a sample input", padding=True, pad_to_multiple_of=8)
                    for key, value in empty_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    normal_tokens = tokenizer("This", pad_to_multiple_of=8, truncation=False, padding=False)
                    for key, value in normal_tokens.items():
                        self.assertNotEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    # Should also work with truncation
                    normal_tokens = tokenizer("This", padding=True, truncation=True, pad_to_multiple_of=8)
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

    def test_tokenizers_common_ids_setters(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                attributes_list = [
                    "bos_token",
                    "eos_token",
                    "unk_token",
                    "sep_token",
                    "pad_token",
                    "cls_token",
                    "mask_token",
                ]

                vocab = tokenizer.get_vocab()
                token_id_to_test_setters = next(iter(vocab.values()))
                token_to_test_setters = tokenizer.convert_ids_to_tokens(
                    token_id_to_test_setters, skip_special_tokens=False
                )
                token_id_to_test_setters = token_id_to_test_setters + tokenizer.image_vocab_size

                for attr in attributes_list:
                    setattr(tokenizer, attr + "_id", None)
                    self.assertEqual(getattr(tokenizer, attr), None)
                    self.assertEqual(getattr(tokenizer, attr + "_id"), None)

                    setattr(tokenizer, attr + "_id", token_id_to_test_setters)
                    self.assertEqual(getattr(tokenizer, attr), token_to_test_setters)
                    self.assertEqual(getattr(tokenizer, attr + "_id"), token_id_to_test_setters)

                setattr(tokenizer, "additional_special_tokens_ids", [])
                self.assertListEqual(getattr(tokenizer, "additional_special_tokens"), [])
                self.assertListEqual(getattr(tokenizer, "additional_special_tokens_ids"), [])

                setattr(tokenizer, "additional_special_tokens_ids", [token_id_to_test_setters])
                self.assertListEqual(getattr(tokenizer, "additional_special_tokens"), [token_to_test_setters])
                self.assertListEqual(getattr(tokenizer, "additional_special_tokens_ids"), [token_id_to_test_setters])

    def test_special_tokens_mask_input_pairs(self):
        pass

    def test_maximum_encoding_length_pair_input(self):
        pass

    def test_mask_output(self):
        pass

    def test_number_of_added_tokens(self):
        pass

    def test_offsets_mapping(self):
        pass
