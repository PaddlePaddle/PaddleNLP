# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import os
import unittest
from paddlenlp.transformers import BertTokenizer, BertJapaneseTokenizer
from paddlenlp.data import Vocab

from common_test import CpuCommonTest
from util import slow, assert_raises
import unittest


class TestBertJapaneseTokenizerFromPretrained(CpuCommonTest):

    @slow
    def test_from_pretrained(self):
        tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese")
        text1 = "こんにちは"
        text2 = "櫓を飛ばす"
        # test batch_encode
        expected_input_ids = [
            2, 10350, 25746, 28450, 3, 20301, 11, 787, 12222, 3, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0
        ]
        expected_token_type_ids = [
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        expected_attention_mask = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        expected_special_tokens_mask = [
            1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
        results = tokenizer([text1], [text2],
                            20,
                            stride=1,
                            pad_to_max_seq_len=True,
                            return_attention_mask=True,
                            return_special_tokens_mask=True)

        self.check_output_equal(results[0]['input_ids'], expected_input_ids)
        self.check_output_equal(results[0]['token_type_ids'],
                                expected_token_type_ids)
        self.check_output_equal(results[0]['attention_mask'],
                                expected_attention_mask)
        self.check_output_equal(results[0]['special_tokens_mask'],
                                expected_special_tokens_mask)
        # test encode
        results = tokenizer(text1, text2, 20, stride=1, pad_to_max_seq_len=True)
        self.check_output_equal(results['input_ids'], expected_input_ids)
        self.check_output_equal(results['token_type_ids'],
                                expected_token_type_ids)

    @slow
    def test_from_pretrained_pad_left(self):
        tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese")
        tokenizer.padding_side = "left"
        text1 = "こんにちは"
        text2 = "櫓を飛ばす"
        # test batch_encode
        expected_input_ids = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 10350, 25746, 28450, 3, 20301, 11,
            787, 12222, 3
        ]
        expected_token_type_ids = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
        ]
        expected_attention_mask = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
        expected_special_tokens_mask = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1
        ]
        results = tokenizer([text1], [text2],
                            20,
                            stride=1,
                            pad_to_max_seq_len=True,
                            return_attention_mask=True,
                            return_special_tokens_mask=True)

        self.check_output_equal(results[0]['input_ids'], expected_input_ids)
        self.check_output_equal(results[0]['token_type_ids'],
                                expected_token_type_ids)
        self.check_output_equal(results[0]['attention_mask'],
                                expected_attention_mask)
        self.check_output_equal(results[0]['special_tokens_mask'],
                                expected_special_tokens_mask)
        # test encode
        results = tokenizer(text1, text2, 20, stride=1, pad_to_max_seq_len=True)
        self.check_output_equal(results['input_ids'], expected_input_ids)
        self.check_output_equal(results['token_type_ids'],
                                expected_token_type_ids)


if __name__ == "__main__":
    unittest.main()
