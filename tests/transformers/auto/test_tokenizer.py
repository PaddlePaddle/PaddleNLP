# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Hugging Face inc.
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

import paddlenlp
from paddlenlp.transformers import AutoTokenizer, is_fast_tokenizer_available


class AutoTokenizerTest(unittest.TestCase):
    def test_fast_tokenizer_import(self):
        tokenizer1 = AutoTokenizer.from_pretrained("__internal_testing__/bert", use_fast=False)
        self.assertIsInstance(tokenizer1, paddlenlp.transformers.BertTokenizer)

        tokenizer2 = AutoTokenizer.from_pretrained("__internal_testing__/bert", use_fast=True)
        if is_fast_tokenizer_available():
            self.assertIsInstance(tokenizer2, paddlenlp.transformers.BertFastTokenizer)
        else:
            self.assertIsInstance(tokenizer2, paddlenlp.transformers.BertTokenizer)

    def test_fast_tokenizer_non_exist(self):
        tokenizer1 = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
        # T5 FastTokenizer doesn't exist yet, so from_pretrained will return the normal tokenizer.
        self.assertIsInstance(tokenizer1, paddlenlp.transformers.T5Tokenizer)

    def test_use_faster(self):
        tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/bert", use_faster=True)
        if is_fast_tokenizer_available():
            self.assertIsInstance(tokenizer, paddlenlp.transformers.BertFastTokenizer)
        else:
            self.assertIsInstance(tokenizer, paddlenlp.transformers.BertTokenizer)
