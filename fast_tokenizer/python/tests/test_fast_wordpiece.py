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

import unittest

from fast_tokenizer import ErnieFastTokenizer
from fast_tokenizer.models import WordPiece, FastWordPiece
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger

logger.logger.setLevel("ERROR")


class TestWordpiece(unittest.TestCase):
    def set_flag(self):
        self.use_fast_wordpiece = False
        self.use_fast_wordpiece_with_pretokenization = False

    def setUp(self):
        self.max_seq_length = 128
        self.wordpiece_tokenizer = AutoTokenizer.from_pretrained("ernie-1.0", use_fast=True)
        ernie_vocab = self.wordpiece_tokenizer.vocab
        self.set_flag()
        self.fast_wordpiece_tokenizer = ErnieFastTokenizer(
            ernie_vocab,
            max_sequence_len=self.max_seq_length,
            use_fast_wordpiece=self.use_fast_wordpiece,
            use_fast_wordpiece_with_pretokenization=self.use_fast_wordpiece_with_pretokenization,
        )
        self.dataset = [example["sentence"] for example in load_dataset("clue", "tnews", splits=["train"])]

    def test_encode(self):
        for sentence in self.dataset:
            wordpiece_result = self.wordpiece_tokenizer(sentence, max_length=self.max_seq_length)
            expected_input_ids = wordpiece_result["input_ids"]
            expected_token_type_ids = wordpiece_result["token_type_ids"]

            fast_wordpiece_result = self.fast_wordpiece_tokenizer.encode(sentence)
            actual_input_ids = fast_wordpiece_result.ids
            actual_token_type_ids = fast_wordpiece_result.type_ids
            self.assertEqual(expected_input_ids, actual_input_ids)
            self.assertEqual(expected_token_type_ids, actual_token_type_ids)

    def test_get_offset_mapping(self):
        for i, sentence in enumerate(self.dataset):
            wordpiece_result = self.wordpiece_tokenizer(
                sentence, max_length=self.max_seq_length, return_offsets_mapping=True
            )
            expected_offset_mapping = wordpiece_result["offset_mapping"]

            fast_wordpiece_result = self.fast_wordpiece_tokenizer.encode(sentence)
            actual_offset_mapping = fast_wordpiece_result.offsets
            self.assertEqual(expected_offset_mapping, actual_offset_mapping)


class TestFastWordpiece(TestWordpiece):
    def set_flag(self):
        self.use_fast_wordpiece = True
        self.use_fast_wordpiece_with_pretokenization = False


class TestFastWordpieceWithPretokenization(TestWordpiece):
    def set_flag(self):
        self.use_fast_wordpiece = True
        self.use_fast_wordpiece_with_pretokenization = True


class TestFromfile(unittest.TestCase):
    def setUp(self):
        self.max_seq_length = 128
        t = AutoTokenizer.from_pretrained("ernie-1.0", use_fast=True)
        self.vocab_file = t.init_kwargs["vocab_file"]

    def test(self):
        WordPiece.from_file(self.vocab_file)
        FastWordPiece.from_file(self.vocab_file)


if __name__ == "__main__":
    unittest.main()
