# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import unittest
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.datasets import load_dataset
from faster_tokenizer import ErnieFasterTokenizer, models

logger.logger.setLevel('ERROR')


class TestWordpiece(unittest.TestCase):

    def set_flag(self):
        self.use_faster_wordpiece = False
        self.use_faster_wordpiece_with_pretokenization = False

    def setUp(self):
        self.max_seq_length = 128
        self.wordpiece_tokenizer = AutoTokenizer.from_pretrained("ernie-1.0")
        ernie_vocab = self.wordpiece_tokenizer.vocab.token_to_idx
        self.set_flag()
        self.faster_wordpiece_tokenizer = ErnieFasterTokenizer(
            ernie_vocab,
            max_sequence_len=self.max_seq_length,
            use_faster_wordpiece=self.use_faster_wordpiece,
            use_faster_wordpiece_with_pretokenization=self.
            use_faster_wordpiece_with_pretokenization)
        self.dataset = [
            example["sentence"]
            for example in load_dataset('clue', 'tnews', splits=['train'])
        ]

    def test_encode(self):
        for sentence in self.dataset:
            wordpiece_result = self.wordpiece_tokenizer(
                sentence, max_length=self.max_seq_length)
            expected_input_ids = wordpiece_result['input_ids']
            expected_token_type_ids = wordpiece_result['token_type_ids']

            faster_wordpiece_result = self.faster_wordpiece_tokenizer.encode(
                sentence)
            actual_input_ids = faster_wordpiece_result.ids
            actual_token_type_ids = faster_wordpiece_result.type_ids
            self.assertEqual(expected_input_ids, actual_input_ids)
            self.assertEqual(expected_token_type_ids, actual_token_type_ids)

    def test_get_offset_mapping(self):
        for i, sentence in enumerate(self.dataset):
            wordpiece_result = self.wordpiece_tokenizer(
                sentence,
                max_length=self.max_seq_length,
                return_offsets_mapping=True)
            expected_offset_mapping = wordpiece_result['offset_mapping']

            faster_wordpiece_result = self.faster_wordpiece_tokenizer.encode(
                sentence)
            actual_offset_mapping = faster_wordpiece_result.offsets
            self.assertEqual(expected_offset_mapping, actual_offset_mapping)


class TestFasterWordpiece(TestWordpiece):

    def set_flag(self):
        self.use_faster_wordpiece = True
        self.use_faster_wordpiece_with_pretokenization = False


class TestFasterWordpieceWithPretokenization(TestWordpiece):

    def set_flag(self):
        self.use_faster_wordpiece = True
        self.use_faster_wordpiece_with_pretokenization = True


if __name__ == "__main__":
    unittest.main()
