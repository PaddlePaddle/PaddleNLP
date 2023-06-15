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

import unittest

from paddlenlp.transformers import RemBertTokenizer

from ..test_tokenizer_common import TokenizerTesterMixin


class TestTokenizationRemBert(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = RemBertTokenizer
    test_offsets = False

    def get_tokenizer(self, **kwargs):
        return self.tokenizer_class.from_pretrained("rembert", **kwargs)

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5):
        output_text = "unwanted, running,running."

        if with_prefix_space:
            output_text = " " + output_text

        output_ids = tokenizer.encode(output_text, return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        return output_text, output_ids

    def test_consecutive_unk_string(self):
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            tokens = [tokenizer.unk_token for _ in range(2)]
            string = tokenizer.convert_tokens_to_string(tokens)
            encoding = tokenizer(
                text=string,
                truncation=True,
                return_offsets_mapping=True,
            )
            self.assertEqual(len(encoding["input_ids"]), 4)
            self.assertEqual(len(encoding["offset_mapping"]), 2)

    def test_pretokenized_inputs(self):
        self.skipTest("not implement yet")
