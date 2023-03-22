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

from paddlenlp.transformers import PegasusChineseTokenizer
from tests.testing_utils import get_tests_dir

from ..test_tokenizer_common import TokenizerTesterMixin

SAMPLE_VOCAB = get_tests_dir("fixtures/vocab.zh.pegasus.txt")


class PegasusTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = PegasusChineseTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()
        tokenizer = PegasusChineseTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs) -> PegasusChineseTokenizer:
        return PegasusChineseTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese", **kwargs)

    def get_input_output_texts(self, tokenizer):
        return ("这是一个测试。", "这是一个测试。")

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "</s>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[-4], "<pad>")
        self.assertEqual(vocab_keys[-5], "</s>")
        self.assertEqual(vocab_keys[158], "v")
        self.assertEqual(len(vocab_keys), 50000)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 50000)

    def test_mask_tokens(self):
        tokenizer = self.get_tokenizer()
        # <mask_1> masks whole sentence while <mask_2> masks single word
        raw_input_str = "<mask_1> 为了确保银行决议的 <mask_2> 流动。"
        desired_result = [2, 7569, 26503, 33094, 10328, 3399, 3, 23514, 179, 1]
        ids = tokenizer([raw_input_str], return_tensors=None).input_ids[0]
        self.assertListEqual(desired_result, ids)

    def test_tokenizer_settings(self):
        tokenizer = self.get_tokenizer()
        # The tracebacks for the following asserts are **better** without messages or self.assertEqual
        assert tokenizer.vocab_size == 50000
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1
        assert tokenizer.offset == 100
        assert tokenizer.unk_token_id == tokenizer.offset == 100
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.model_max_length == 1024
        raw_input_str = "确保银行决议的顺利进行。"
        desired_result = [26503, 33094, 10328, 3399, 5396, 612, 4921, 4503, 179, 1]
        ids = tokenizer([raw_input_str], return_tensors=None).input_ids[0]
        self.assertListEqual(desired_result, ids)
        assert tokenizer.convert_ids_to_tokens([0, 1, 2, 3], skip_special_tokens=False) == [
            "<pad>",
            "</s>",
            "<mask_1>",
            "<mask_2>",
        ]

    def test_seq2seq_truncation(self):
        tokenizer = self.get_tokenizer()
        src_texts = ["这将是一个很长很长的文本。" * 150, "short example"]
        tgt_texts = ["这个不是很长但是超过5个字。", "tiny"]
        batch = tokenizer(text=src_texts, padding=True, truncation=True, return_tensors="pd")
        targets = tokenizer(text=tgt_texts, max_length=5, padding=True, truncation=True, return_tensors="pd")

        assert batch.input_ids.shape == [2, 1024]
        assert batch.attention_mask.shape == [2, 1024]
        assert targets["input_ids"].shape == [2, 5]
        assert len(batch) == 2  # input_ids, attention_mask.

    def test_consecutive_unk_string(self):
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            tokens = [tokenizer.unk_token for _ in range(2)]
            string = tokenizer.convert_tokens_to_string(tokens)
            encoding = tokenizer(
                text=string,
                runcation=True,
                return_offsets_mapping=True,
            )
            # BOS is never used.
            self.assertEqual(len(encoding["input_ids"]), 3)
            self.assertEqual(len(encoding["offset_mapping"]), 3)
