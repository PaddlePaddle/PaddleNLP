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

import json
import os
import unittest

from paddlenlp.transformers import CTRLTokenizer

# from paddlenlp.transformers import CodeGenTokenizer
from paddlenlp.transformers.codegen.tokenizer import VOCAB_FILES_NAMES

# from ...testing_utils import slow
from ..test_tokenizer_common import TokenizerTesterMixin


class CTRLTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = CTRLTokenizer
    test_rust_tokenizer = False
    test_seq2seq = False

    def setUp(self):
        super().setUp()
        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ["adapt", "re@@", "a@@", "apt", "c@@", "t", "<unk>"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "a p", "ap t</w>", "r e", "a d", "ad apt</w>", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return CTRLTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "adapt react readapt apt"
        output_text = "adapt react readapt apt"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = CTRLTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "adapt react readapt apt"
        bpe_tokens = "adapt re@@ a@@ c@@ t re@@ adapt apt".split()
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]

        input_bpe_tokens = [0, 1, 2, 4, 5, 1, 0, 3, 6]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_add_special_tokens(self):
        pass

    def test_add_tokens(self):
        pass

    def test_add_tokens_tokenizer(self):
        pass

    def test_added_token_are_matched_longest_first(self):
        pass

    def test_added_tokens_do_lower_case(self):
        pass

    def test_consecutive_unk_string(self):
        pass

    def test_encode_decode_with_spaces(self):
        pass

    def test_offsets_mapping_with_unk(self):
        pass

    def test_pretokenized_inputs(self):
        pass

    def test_pretrained_model_lists(self):
        pass

    def test_tokenize_special_tokens(self):
        pass

    def test_save_and_load_tokenizer(self):
        pass

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        pass
