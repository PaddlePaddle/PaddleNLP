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

import os
import unittest

from fast_tokenizer import ClipFastTokenizer, models
from paddlenlp.utils.downloader import get_path_from_url


class TestClipFastTokenizer(unittest.TestCase):
    def setUp(self):
        vocab_path = os.path.join(os.getcwd(), "vocab.json")
        merges_path = os.path.join(os.getcwd(), "merges.txt")
        if not os.path.exists(vocab_path):
            get_path_from_url(
                "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-vit-large-patch14/vocab.json", os.getcwd()
            )
        if not os.path.exists(merges_path):
            get_path_from_url(
                "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-vit-large-patch14/merges.txt", os.getcwd()
            )
        vocab, merges = models.BPE.read_file(vocab_path, merges_path)
        self.tokenizer = ClipFastTokenizer(vocab, merges)
        self.expected_ids = [
            49406,
            320,
            1342,
            272,
            272,
            335,
            273,
            273,
            274,
            16368,
            13439,
            2971,
            748,
            531,
            13610,
            323,
            1896,
            8445,
            323,
            539,
            320,
            2368,
            49407,
        ]
        self.expected_tokens = [
            "<|startoftext|>",
            "a</w>",
            "'ll</w>",
            "1</w>",
            "1</w>",
            "p</w>",
            "2</w>",
            "2</w>",
            "3</w>",
            "rf</w>",
            "âĺĨ</w>",
            "ho</w>",
            "!!</w>",
            "to</w>",
            "?'</w>",
            "d</w>",
            "'d</w>",
            "''</w>",
            "d</w>",
            "of</w>",
            "a</w>",
            "cat</w>",
            "<|endoftext|>",
        ]
        self.input_text = "A\n'll 11p223RF☆ho!!to?'d'd''d of a cat"

    def test_encode(self):
        result = self.tokenizer.encode(self.input_text)
        self.assertEqual(result.tokens, self.expected_tokens)
        self.assertEqual(result.ids, self.expected_ids)
