# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from paddlenlp.transformers import XLMRobertaTokenizer

from ..test_tokenizer_common import TokenizerTesterMixin

# VOCAB_FILES_NAMES = XLMRobertaTokenizer.resource_files_names


class XLMRobertaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    test_offsets = False
    tokenizer_class = XLMRobertaTokenizer

    # Set up method called before each test
    def setUp(self):
        super().setUp()
        self.vocab_file = "BAAI/bge-m3"
        self.special_tokens_map = {"unk_token": "<unk>"}

    # Method to get a tokenizer instance with specified keyword arguments
    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return XLMRobertaTokenizer.from_pretrained(self.vocab_file, **kwargs)

    # Test method to check tokenization
    def test_tokenization(self):
        tokenizer = self.get_tokenizer()
        text = "Hello, how are you?"
        tokens = tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    # Test method to check conversion of token to ID
    def test_token_to_id(self):
        tokenizer = self.get_tokenizer()
        token = "Hello"
        token_id = tokenizer.convert_tokens_to_ids(token)
        self.assertIsInstance(token_id, int)

    # Test method to check conversion of ID to token
    def test_id_to_token(self):
        tokenizer = self.get_tokenizer()
        token_id = tokenizer.convert_tokens_to_ids("How")
        token = tokenizer.convert_ids_to_tokens(token_id)
        self.assertEqual(token, "How")

    # Test method to check special tokens
    def test_special_tokens(self):
        tokenizer = self.get_tokenizer(
            vocab_file=self.vocab_file, cls_token="<cls>", sep_token="<sep>", pad_token="<pad>"
        )
        self.assertEqual(tokenizer.cls_token, "<cls>")
        self.assertEqual(tokenizer.sep_token, "<sep>")
        self.assertEqual(tokenizer.pad_token, "<pad>")

    # Test method to check building inputs with special tokens
    def test_build_inputs_with_special_tokens(self):
        tokenizer = self.get_tokenizer()
        token_ids_0 = tokenizer.convert_tokens_to_ids(["Hello", "world"])
        token_ids_1 = tokenizer.convert_tokens_to_ids(["How", "are", "you"])

        input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
        self.assertEqual(input_ids[0], tokenizer.cls_token_id)
        self.assertEqual(input_ids[-1], tokenizer.sep_token_id)


if __name__ == "__main__":
    unittest.main()
