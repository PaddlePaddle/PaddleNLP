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
import sys
import json

from paddlenlp.transformers import RobertaTokenizer
from paddlenlp.data import Vocab

from common_test import CpuCommonTest
from util import slow, assert_raises
import unittest


class TestRobertaTokenizer(CpuCommonTest):

    def set_attr(self):
        self.do_lower_case = True

    def create_input_file(self):
        vocab = [
            "[UNK]", "[CLS]", "[SEP]", "th", "##is", "is", "simple", "text",
            "a", "an", "for", "easy", "which", "children", "[MASK]", "[PAD]"
        ]
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        vocab_file = os.path.join(curr_dir, "vocab.txt")
        with open(vocab_file, "w") as fw:
            for v in vocab:
                fw.write(v)
                fw.write('\n')
        self.vocab_file = vocab_file
        self.vocab = vocab

    def set_test_case(self):
        self.text = "this is a simple text"
        self.expected_text_array = ['th', '##is', 'is', 'a', 'simple', 'text']

    def setUp(self):
        self.set_attr()
        self.create_input_file()
        self.set_test_case()
        self.tokenizer = RobertaTokenizer(vocab_file=self.vocab_file,
                                          merges_file=None,
                                          do_lower_case=self.do_lower_case)

    def test_tokenize(self):
        text_array = self.tokenizer.tokenize(self.text)
        self.check_output_equal(text_array, self.expected_text_array)
        self.check_output_equal(
            self.tokenizer.convert_tokens_to_string(text_array), self.text)

    def test_call(self):
        expected_input_ids = [1, 3, 4, 5, 8, 6, 7, 2]
        expected_token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0]
        expected_attention_mask = [1] * len(expected_input_ids)
        expected_tokens_mask = [1, 0, 0, 0, 0, 0, 0, 1]
        result = self.tokenizer("This  is a simple text",
                                return_attention_mask=True,
                                return_length=True,
                                return_special_tokens_mask=True)
        self.check_output_equal(result['input_ids'], expected_input_ids)
        self.check_output_equal(result['token_type_ids'],
                                expected_token_type_ids)
        self.check_output_equal(result['seq_len'], len(expected_token_type_ids))
        self.check_output_equal(result['attention_mask'],
                                expected_attention_mask)
        self.check_output_equal(result['special_tokens_mask'],
                                expected_tokens_mask)

    def test_call_pair(self):
        expected_input_ids = [1, 3, 4, 5, 8, 6, 7, 2, 12, 5, 11, 10, 13, 2]
        expected_token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        result = self.tokenizer("This is a simple text",
                                "which is easy for children")
        self.check_output_equal(result['input_ids'], expected_input_ids)
        self.check_output_equal(result['token_type_ids'],
                                expected_token_type_ids)

    def test_call_batch(self):
        expected_input_ids = [1, 3, 4, 5, 8, 6, 7, 2]
        expected_token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0]
        results = self.tokenizer(
            ["This is a simple text", "this Is A    simple text"])
        for result in results:
            self.check_output_equal(result['input_ids'], expected_input_ids)
            self.check_output_equal(result['token_type_ids'],
                                    expected_token_type_ids)

    def test_call_truncate_seq(self):
        expected_input_ids = [1, 3, 2, 3, 2]
        expected_token_type_ids = [0, 0, 0, 1, 1]
        results = self.tokenizer("This is a simple text",
                                 "this Is A    simple text", 5)

        self.check_output_equal(results['input_ids'], expected_input_ids)
        self.check_output_equal(results['token_type_ids'],
                                expected_token_type_ids)

    # Test PretrainedTokenizer
    def test_truncate_only_first(self):
        ids = [1, 3, 4, 5, 8, 6, 7, 2]
        pair_ids = [12, 5, 11, 10, 13, 2]

        truncate_ids, truncate_pair_ids, _ = self.tokenizer.truncate_sequences(
            ids, pair_ids, 3, truncation_strategy='only_first')
        self.check_output_equal(truncate_ids, ids[:-3])
        self.check_output_equal(truncate_pair_ids, pair_ids)

    def test_truncate_only_second(self):
        ids = [1, 3, 4, 5, 8, 6, 7, 2]
        pair_ids = [12, 5, 11, 10, 13, 2]

        truncate_ids, truncate_pair_ids, _ = self.tokenizer.truncate_sequences(
            ids, pair_ids, 3, truncation_strategy='only_second')
        self.check_output_equal(truncate_ids, ids)
        self.check_output_equal(truncate_pair_ids, pair_ids[:-3])

    @assert_raises(ValueError)
    def test_truncate_do_not_truncate(self):
        ids = [1, 3, 4, 5, 8, 6, 7, 2]
        pair_ids = [12, 5, 11, 10, 13, 2]
        truncate_ids, truncate_pair_ids, _ = self.tokenizer.truncate_sequences(
            ids, pair_ids, 3, truncation_strategy='do_not_truncate')

    @assert_raises(ValueError)
    def test_truncate_error_strategy(self):
        ids = [1, 3, 4, 5, 8, 6, 7, 2]
        pair_ids = [12, 5, 11, 10, 13, 2]
        truncate_ids, truncate_pair_ids, _ = self.tokenizer.truncate_sequences(
            ids, pair_ids, 1, truncation_strategy='')

    def test_save_pretrained(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(curr_dir, "pretrained_model")
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self.tokenizer.save_pretrained(model_path)

        vocab_path = os.path.join(
            model_path, self.tokenizer.resource_files_names['vocab_file'])
        with open(vocab_path, "r") as fr:
            vocabs = [vocab.strip() for vocab in fr.readlines()]
        self.check_output_equal(vocabs, self.vocab)

    @assert_raises(RuntimeError)
    def test_from_pretrained_non_exist(self):
        RobertaTokenizer.from_pretrained("")

    def test_vocab_size(self):
        self.check_output_equal(self.tokenizer.vocab_size, len(self.vocab))

    def test_all_special_tokens(self):
        expected_special_tokens_set = set([
            self.tokenizer.pad_token, self.tokenizer.mask_token,
            self.tokenizer.cls_token, self.tokenizer.unk_token,
            self.tokenizer.sep_token
        ])
        self.check_output_equal(set(self.tokenizer.all_special_tokens),
                                expected_special_tokens_set)
        self.check_output_equal(set(self.tokenizer.all_special_ids),
                                set([0, 1, 2, 14, 15]))

    @assert_raises(ValueError)
    def test_non_exist_vocab_file(self):
        RobertaTokenizer("non_exist.txt", merges_file=None)


class TestRobertaTokenizerFromPretrained(CpuCommonTest):

    @slow
    def test_from_pretrained(self):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-wwm-ext")
        text1 = "这是一个简单文本"
        text2 = "小孩子都看得懂"
        # test batch_encode
        expected_input_ids = [
            101, 6821, 3221, 671, 702, 5042, 1296, 3152, 3315, 102, 2207, 2111,
            2094, 6963, 4692, 2533, 2743, 102, 0, 0
        ]
        expected_token_type_ids = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
        ]
        expected_attention_mask = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
        ]
        expected_special_tokens_mask = [
            1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1
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
        tokenizer = RobertaTokenizer.from_pretrained("roberta-wwm-ext")
        tokenizer.padding_side = "left"
        text1 = "这是一个简单文本"
        text2 = "小孩子都看得懂"
        # test batch_encode
        expected_input_ids = [
            0, 0, 101, 6821, 3221, 671, 702, 5042, 1296, 3152, 3315, 102, 2207,
            2111, 2094, 6963, 4692, 2533, 2743, 102
        ]
        expected_token_type_ids = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1
        ]
        expected_attention_mask = [
            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
        expected_special_tokens_mask = [
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1
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
