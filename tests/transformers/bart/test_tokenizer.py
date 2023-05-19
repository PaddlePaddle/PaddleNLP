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
import json
import os
import unittest

from paddlenlp.transformers import BartTokenizer

from ..test_tokenizer_common import TokenizerTesterMixin, filter_roberta_detectors

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


class TestTokenizationBart(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = BartTokenizer
    test_rust_tokenizer = False
    test_offsets = False
    from_pretrained_filter = filter_roberta_detectors

    def setUp(self):
        super().setUp()
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
            "<mask>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "cls_token": "<s>",
            "sep_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
        }

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return "lower newer", "lower newer"

    def default_tokenizer(self):
        return BartTokenizer.from_pretrained("bart-large")

    def test_prepare_batch(self):
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        expected_src_tokens = [0, 250, 251, 17818, 13, 39186, 1938, 4, 2]

        for tokenizer in [BartTokenizer.from_pretrained("bart-large")]:
            batch = tokenizer(
                text=src_text,
                max_length=len(expected_src_tokens),
                padding=True,
                return_attention_mask=True,
                return_tensors="pd",
            )
            self.assertEqual([2, 9], batch.input_ids.shape)
            self.assertEqual([2, 9], batch.attention_mask.shape)
            result = batch.input_ids.tolist()[0]
            self.assertListEqual(expected_src_tokens, result)
            # Test that special tokens are reset

    def test_prepare_batch_empty_target_text(self):
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        for tokenizer in [BartTokenizer.from_pretrained("bart-large")]:
            batch = tokenizer(text=src_text, padding=True, return_tensors="pd", return_attention_mask=True)
            # check if input_ids are returned and no labels
            self.assertIn("input_ids", batch)
            self.assertIn("attention_mask", batch)
            self.assertNotIn("labels", batch)
            self.assertNotIn("decoder_attention_mask", batch)

    def test_tokenizer_as_target_length(self):
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        for tokenizer in [BartTokenizer.from_pretrained("bart-large")]:
            targets = tokenizer(text=tgt_text, max_length=32, padding="max_length", return_tensors="pd")
            self.assertEqual(32, targets["input_ids"].shape[1])

    def test_prepare_batch_not_longer_than_maxlen(self):
        for tokenizer in [BartTokenizer.from_pretrained("bart-large", max_len=1024)]:
            batch = tokenizer(
                text=["I am a small frog" * 1024, "I am a small frog"],
                padding=True,
                truncation=True,
                return_tensors="pd",
            )
            self.assertEqual(batch.input_ids.shape, [2, 1024])

    def test_special_tokens(self):

        src_text = ["A long paragraph for summarization."]
        tgt_text = [
            "Summary of the text.",
        ]
        for tokenizer in [BartTokenizer.from_pretrained("bart-large")]:
            inputs = tokenizer(text=src_text, return_tensors="pd")
            targets = tokenizer(text=tgt_text, return_tensors="pd")
            input_ids = inputs["input_ids"]
            labels = targets["input_ids"]
            self.assertTrue((input_ids[:, 0] == tokenizer.bos_token_id).all().item())
            self.assertTrue((labels[:, 0] == tokenizer.bos_token_id).all().item())
            self.assertTrue((input_ids[:, -1] == tokenizer.eos_token_id).all().item())
            self.assertTrue((labels[:, -1] == tokenizer.eos_token_id).all().item())

    def test_pretokenized_inputs(self):
        pass
