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
import paddle
from paddlenlp.transformers import BigBirdTokenizer

from common_test import CpuCommonTest


class TestBigBirdTokenizer(CpuCommonTest):
    def set_input(self):
        self.max_seq_len = 40
        self.max_pred_len = 3

    def set_output(self):
        self.expected_span_ids = np.array([
            65, 1153, 36677, 3766, 2747, 427, 3830, 419, 530, 16474, 1677, 6464,
            5441, 385, 7002, 363, 2099, 387, 5065, 441, 484, 2375, 3583, 5682,
            16812, 474, 34179, 1266, 8951, 391, 34478, 871, 67, 67, 385, 29223,
            2447, 388, 635, 66
        ])
        self.expected_masked_lm_positions = np.array([2, 32, 33])
        self.expected_masked_lm_ids = np.array([4558, 2757, 15415])
        self.expected_masked_lm_weights = np.array([1., 1., 1.])

    def setUp(self):
        self.text = 'An extremely powerful film that certainly isnt '\
            'appreciated enough Its impossible to describe the experience '\
            'of watching it The recent UK television adaptation was shameful  '\
            'too ordinary and bland This original manages to imprint itself '\
            'in your memory'
        self.tokenizer = BigBirdTokenizer.from_pretrained(
            'bigbird-base-uncased')
        self.set_input()
        self.set_output()

    def test_vocab_size(self):
        self.check_output_equal(self.tokenizer.vocab_size, 50358)

    def test_tokenize(self):
        np.random.seed(102)
        result = self.tokenizer.encode(self.text, self.max_seq_len,
                                       self.max_pred_len)
        span_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = result
        self.check_output_equal(span_ids, self.expected_span_ids)
        self.check_output_equal(masked_lm_positions,
                                self.expected_masked_lm_positions)
        self.check_output_equal(masked_lm_ids, self.expected_masked_lm_ids)
        self.check_output_equal(masked_lm_weights,
                                self.expected_masked_lm_weights)


class TestBigBirdTokenizerLongMaxPredLen(TestBigBirdTokenizer):
    def set_input(self):
        self.max_seq_len = 40
        self.max_pred_len = 8

    def set_output(self):
        self.expected_span_ids = np.array([
            65, 1153, 48226, 3766, 2747, 427, 67, 419, 530, 16474, 1677, 6464,
            5441, 385, 7002, 363, 2099, 387, 5065, 441, 484, 67, 3583, 5682,
            16812, 474, 34179, 1266, 8951, 391, 34478, 871, 34299, 67, 385,
            29223, 2447, 67, 635, 66
        ])
        self.expected_masked_lm_positions = np.array(
            [2, 6, 21, 32, 33, 37, 0, 0])
        self.expected_masked_lm_ids = np.array(
            [4558, 3830, 2375, 2757, 15415, 388, 0, 0])
        self.expected_masked_lm_weights = np.array(
            [1., 1., 1., 1., 1., 1., 0., 0.])
