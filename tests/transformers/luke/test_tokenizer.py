# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import shutil
import tempfile
import unittest

from paddlenlp.transformers import LukeTokenizer

from ..test_tokenizer_common import TokenizerTesterMixin

VOCAB_FILES_NAMES = LukeTokenizer.resource_files_names


class TestTokenizationLuke(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = LukeTokenizer
    test_offsets = False

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
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
            "</s>",
            "<pad>",
            "<s>",
            "<mask>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        entity_vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[MASK2]": 3}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        self.entity_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["entity_file"])

        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))
        with open(self.entity_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(entity_vocab))

    def test_add_special_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, ids = self.get_clean_sequence(tokenizer)

                special_token = "[SPECIAL_TOKEN]"

                tokenizer.add_special_tokens({"additional_special_tokens": special_token})
                encoded_special_token = tokenizer.encode(
                    special_token, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                self.assertEqual(len(encoded_special_token), len(special_token))

                text = tokenizer.decode(ids + encoded_special_token, clean_up_tokenization_spaces=False)
                encoded = tokenizer.encode(text, return_token_type_ids=None, add_special_tokens=False)["input_ids"]

                input_encoded = tokenizer.encode(input_text, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]
                special_token_id = tokenizer.encode(
                    special_token, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                self.assertEqual(encoded, input_encoded + special_token_id)

                decoded = tokenizer.decode(encoded, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_tokenize_special_tokens(self):
        """Test `tokenize` with special tokens."""
        tokenizers = self.get_tokenizers(do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                SPECIAL_TOKEN_1 = "[SPECIAL_TOKEN_1]"
                SPECIAL_TOKEN_2 = "[SPECIAL_TOKEN_2]"
                tokenizer.add_tokens([SPECIAL_TOKEN_1], special_tokens=True)
                tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKEN_2]})

                token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
                token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)

                self.assertEqual(len(token_1), len(SPECIAL_TOKEN_1))
                self.assertEqual(len(token_2), 1)
                self.assertEqual(token_1[0], SPECIAL_TOKEN_1[0])
                self.assertEqual(token_2[0], SPECIAL_TOKEN_2)

    def test_consecutive_unk_string(self):
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            tokens = [tokenizer.unk_token for _ in range(2)]
            string = tokenizer.convert_tokens_to_string(tokens)
            encoding = tokenizer(
                text=string,
            )
            self.assertEqual(len(encoding["input_ids"]), 4)

    def test_save_and_load_tokenizer(self):
        # safety check on max_len default value so we are sure the test works
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertNotEqual(tokenizer.model_max_length, 42)

        # Now let's start the test
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens["input_ids"], after_tokens["input_ids"])
                self.assertEqual(before_vocab.keys(), after_vocab.keys())

                shutil.rmtree(tmpdirname)

    def test_conversion_reversible(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab = tokenizer.get_vocab()
                for word, ind in vocab.items():
                    if word == tokenizer.unk_token:
                        continue
                    self.assertEqual(tokenizer.encoder[word], ind)
                    self.assertEqual(tokenizer.convert_ids_to_tokens(ind), word)

    def test_call(self):
        self.skipTest("Direct call is not the same as encode")

    def test_tokenizers_common_ids_setters(self):
        self.skipTest("Add token not implement yet")

    def test_add_tokens(self):
        self.skipTest("Add token not implement yet")

    def test_add_tokens_tokenizer(self):
        self.skipTest("Add token not implement yet")

    def test_added_token_serializable(self):
        self.skipTest("Add token not implement yet")

    def test_added_tokens_do_lower_case(self):
        self.skipTest("Add token not implement yet")

    def test_added_token_are_matched_longest_first(self):
        self.skipTest("Add token not implement yet")

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        self.skipTest("Add token not implement yet")

    def test_encode_decode_with_spaces(self):
        self.skipTest("Add token not implement yet")
