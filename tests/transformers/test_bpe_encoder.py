# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 HuggingFace Inc.
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

import inspect
import itertools
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
import unittest
from collections import OrderedDict
from itertools import takewhile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from paddlenlp.transformers import (AlbertTokenizer, AutoTokenizer,
                                    BertTokenizer, PretrainedTokenizer)
from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase
from paddlenlp.transformers.tokenizer_utils import AddedToken, Trie, BpeEncoder, BPETokenizer

from tests.testing_utils import get_tests_dir, slow
from tests.transformers.test_tokenizer_common import TokenizerTesterMixin

sys.path.append(str(Path(__file__).parent.parent / "utils"))

NON_ENGLISH_TAGS = [
    "chinese", "dutch", "french", "finnish", "german", "multilingual"
]

SMALL_TRAINING_CORPUS = [
    ["This is the first sentence.", "This is the second one."],
    [
        "This sentence (contains #) over symbols and numbers 12 3.",
        "But not this one."
    ],
]


def filter_non_english(_, pretrained_name: str):
    """Filter all the model for non-english language"""
    return not any([lang in pretrained_name for lang in NON_ENGLISH_TAGS])


class BPEEncoderTest(unittest.TestCase):

    def setUp(self):
        vocab_file = get_tests_dir("fixtures/bpe.en/vocab.json")
        merges_file = get_tests_dir("fixtures/bpe.en/merges.txt")

        self.encoder = BpeEncoder(vocab_file=vocab_file,
                                  merges_file=merges_file)

    def test_tokenizer(self):
        text = " lower newer"
        bpe_tokens = ["\u0120low", "er", "\u0120", "n", "e", "w", "er"]
        tokens = self.encoder._tokenize(text)

        self.assertListEqual(tokens, bpe_tokens)

        decoded_text = self.encoder.convert_tokens_to_string(tokens)
        self.assertEqual(text, decoded_text)

    def test_tokenizer_encode_decode(self):
        text = " lower newer"
        bpe_tokens = ["\u0120low", "er", "\u0120", "n", "e", "w", "er"]
        token_ids = self.encoder.encode(text)
        tokens = [self.encoder.decoder[token_id] for token_id in token_ids]

        self.assertListEqual(tokens, bpe_tokens)

        decoded_text = self.encoder.decode(token_ids)
        self.assertEqual(text, decoded_text)

    def test_unk_word(self):
        text = " lower newer a"
        with self.assertRaises(KeyError):
            self.encoder.encode(text)

        # can tokenize correct
        tokens = self.encoder._tokenize(text)

        # recognize the `a` as the <unk-token>
        token_ids = [
            self.encoder._convert_token_to_id(token) for token in tokens
        ]

        decoded_tokens = [
            self.encoder._convert_id_to_token(token_id)
            for token_id in token_ids
        ]
        self.assertIn(self.encoder.unk_token, decoded_tokens)


class BPETokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    only_english_character = False
    tokenizer_class = BPETokenizer

    def setUp(self):

        self.vocab_file = get_tests_dir("fixtures/bpe.en/vocab.json")
        self.merges_file = get_tests_dir("fixtures/bpe.en/merges.txt")

        self.tmpdirname = tempfile.mkdtemp()
        tokenizers_list = [(BPETokenizer, get_tests_dir("fixtures/bpe.en"), {})]
        self.tokenizers_list = tokenizers_list[:1]

    def get_tokenizer(self, **kwargs):
        return BPETokenizer.from_pretrained(get_tests_dir("fixtures/bpe.en"),
                                            **kwargs)

    def test_pretokenized_inputs(self):
        # Test when inputs are pretokenized

        tokenizers = self.get_tokenizers(
            do_lower_case=False)  # , add_prefix_space=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                if hasattr(
                        tokenizer,
                        "add_prefix_space") and not tokenizer.add_prefix_space:
                    continue

                # Prepare a sequence from our tokenizer vocabulary
                sequence, ids = self.get_clean_sequence(tokenizer,
                                                        with_prefix_space=True,
                                                        max_length=20)
                # sequence_no_prefix_space = sequence.strip()
                token_sequence = sequence.split()
                # Test encode for pretokenized inputs
                output_sequence = tokenizer.encode(
                    sequence,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                self.assertEqual(ids, output_sequence)

    def test_pretrained_model_lists(self):
        self.skipTest("skip this empty bpe tokenizer")

    def test_offsets_mapping(self):
        self.skipTest(
            "using basic-tokenizer or word-piece tokenzier to do this test, so to skip this unittest"
        )
