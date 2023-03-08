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

from paddlenlp.transformers.xlm.tokenizer import XLMTokenizer
from tests.testing_utils import get_tests_dir, slow

from ..test_tokenizer_common import TokenizerTesterMixin


class XLMTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = XLMTokenizer
    test_fast_tokenizer = False

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
            "w</w>",
            "r</w>",
            "t</w>",
            "lo",
            "low",
            "er</w>",
            "low</w>",
            "lowest</w>",
            "newer</w>",
            "wider</w>",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["l o 123", "lo w 1456", "e r</w> 1789", ""]

        self.vocab_file = os.path.join(self.tmpdirname, XLMTokenizer.resource_files_names["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, XLMTokenizer.resource_files_names["merges_file"])
        with open(self.vocab_file, "w") as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(self.merges_file, "w") as fp:
            fp.write("\n".join(merges))

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "l")
        self.assertEqual(vocab_keys[-1], "<special9>")
        self.assertEqual(len(vocab_keys), 34)

    def test_add_tokens_tokenizer(self):
        pass

    def test_added_token_serializable(self):
        pass

    def test_consecutive_unk_string(self):
        pass

    def test_offsets_mapping(self):
        pass

    def test_save_and_load_tokenizer(self):
        pass

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        pass

    def test_add_tokens(self):
        tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-en-2048")
        vocab_size = len(tokenizer)
        self.assertEqual(tokenizer.add_tokens(""), 0)
        self.assertEqual(tokenizer.add_tokens("testoken"), 1)
        self.assertEqual(tokenizer.add_tokens(["testoken1", "testtoken2"]), 2)
        self.assertEqual(len(tokenizer.get_vocab()), vocab_size + 3)

    def test_full_tokenizer(self):
        """Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt"""
        tokenizer = XLMTokenizer(self.vocab_file, self.merges_file)

        text = "lower"
        bpe_tokens = ["low", "er</w>"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + ["<unk>"]
        input_bpe_tokens = [14, 15, 20]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    @slow
    def test_sequence_builders(self):
        tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-en-2048")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [0] + text + [1]
        assert encoded_pair == [0] + text + [1] + text_2 + [1]
