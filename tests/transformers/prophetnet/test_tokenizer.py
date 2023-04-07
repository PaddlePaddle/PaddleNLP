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
import os
import shutil
import tempfile
import unittest
import warnings

from paddlenlp.transformers import ProphetNetTokenizer

from ..test_tokenizer_common import TokenizerTesterMixin

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
}


class TestTokenizationProphetNet(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = ProphetNetTokenizer
    test_rust_tokenizer = False
    test_offsets = False

    def setUp(self):
        super().setUp()
        vocab = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.special_tokens_map = {
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "bos_token": "[SEP]",
            "eos_token": "[SEP]",
            "cls_token": "[CLS]",
            "x_sep_token": "[X_SEP]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        }
        self.vocab_file = os.path.join(self.tmpdirname, ProphetNetTokenizer.resource_files_names["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def test_save_and_load_tokenizer(self):
        warnings.warn("Every addtoken not in vocab is unk_token")
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertNotEqual(tokenizer.model_max_length, 42)

        # Now let's start the test
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens["input_ids"], after_tokens["input_ids"])
                self.assertDictEqual(before_vocab, after_vocab)

                shutil.rmtree(tmpdirname)

    def test_add_tokens_tokenizer(self):
        warnings.warn("Every token not in vocab is unk_token")
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode(
                    "aaaaa bbbbbb low cccccccccdddddddd l", return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                self.assertGreaterEqual(len(tokens), 4)
                self.assertEqual(tokens[0], tokenizer.unk_token_id)
                self.assertEqual(tokens[0], tokenizer.unk_token_id)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l",
                    return_token_type_ids=None,
                    add_special_tokens=False,
                )["input_ids"]

                self.assertGreaterEqual(len(tokens), 6)
                self.assertEqual(tokens[0], tokenizer.unk_token_id)
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00E9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def test_encode_decode_with_spaces(self):
        self.skipTest("Every token not in vocab is unk_token")

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        self.skipTest("Every token not in vocab is unk_token")

    def test_consecutive_unk_string(self):
        self.skipTest("Every token not in vocab is unk_token")

    def test_pretokenized_inputs(self):
        self.skipTest("tokenizer is_split_into_words not implement yet")
