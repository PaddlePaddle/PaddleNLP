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
import unittest
from typing import List
import shutil

import sentencepiece as spm
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizerBase, PretrainedTokenizer
from paddlenlp.transformers.ernie_m.tokenizer import ErnieMTokenizer
from paddlenlp.transformers.tokenizer_utils import _is_whitespace, _is_control, _is_punctuation

from ...testing_utils import slow, get_tests_dir
from ..test_tokenizer_common import TokenizerTesterMixin, filter_non_english

EN_SENTENCEPIECE = get_tests_dir("fixtures/test_sentencepiece_bpe.model")
EN_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe.vocab.txt")


class ErnieMEnglishTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = ErnieMTokenizer
    space_between_special_tokens = True

    def setUp(self):
        super().setUp()

        tokenizer = ErnieMTokenizer(vocab_file=EN_VOCAB, sentencepiece_model_file=EN_SENTENCEPIECE, unk_token="<unk>")
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        return ErnieMTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "This is a test"
        output_text = "This is a test"
        return input_text, output_text

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<unk>"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_full_tokenizer(self):
        tokenizer = ErnieMTokenizer.from_pretrained(self.tmpdirname)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [474, 97, 5, 3, 263])

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                "▁I",
                "▁was",
                "▁b",
                "or",
                "n",
                "▁in",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                "▁and",
                "▁this",
                "▁is",
                "▁f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids, [16, 52, 12, 27, 936, 39, 0, 998, 992, 992, 992, 953, 32, 119, 97, 20, 81, 939, 0, 951]
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                "▁I",
                "▁was",
                "▁b",
                "or",
                "n",
                "▁in",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                "▁and",
                "▁this",
                "▁is",
                "▁f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual(
            [tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]], [["▁T", "est"], ["\xad"], ["▁t", "est"]]
        )

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("ernie-m-base")

        text = tokenizer.encode("sequence builders", return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build", return_token_type_ids=None, add_special_tokens=False)[
            "input_ids"
        ]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [
            tokenizer.sep_token_id,
            tokenizer.sep_token_id,
        ] + text_2 + [tokenizer.sep_token_id]

    def test_token_type_ids(self):
        self.skipTest("Ernie-M model doesn't have token_type embedding. so skip this test")
