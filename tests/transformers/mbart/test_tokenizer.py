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

import inspect
import tempfile
import unittest

from paddlenlp.transformers import SPIECE_UNDERLINE
from paddlenlp.transformers.mbart.modeling import shift_tokens_right
from paddlenlp.transformers.mbart.tokenizer import MBartTokenizer

from ...testing_utils import get_tests_dir
from ..test_tokenizer_common import TokenizerTesterMixin

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

EN_CODE = 250004
RO_CODE = 250026


class MBartTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MBartTokenizer
    test_sentencepiece = True

    test_offsets = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = MBartTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = MBartTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [value + tokenizer.fairseq_offset for value in [285, 46, 10, 170, 382]],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [
                value + tokenizer.fairseq_offset
                for value in [8, 21, 84, 55, 24, 19, 7, 2, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 2, 4]
                #                                       ^ unk: 2 + 1 = 3                  unk: 2 + 1 = 3 ^
            ],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )

    def test_tokenizer_slow_store_full_signature(self):
        signature = inspect.signature(self.tokenizer_class.__init__)
        tokenizer = self.get_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty:
                if parameter_name != "mbart_type":
                    self.assertIn(parameter_name, tokenizer.init_kwargs)


class MBartEnroIntegrationTest(unittest.TestCase):
    checkpoint_name = "mbart-large-en-ro"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        """ Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for Syria is that "there is no military solution" to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people.""",
    ]
    tgt_text = [
        "Şeful ONU declară că nu există o soluţie militară în Siria",
        "Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al Rusiei"
        ' pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu vor'
        " face decât să înrăutăţească violenţele şi mizeria pentru milioane de oameni.",
    ]
    expected_src_tokens = [8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53, 187895, 23, 51712, 2, EN_CODE]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: MBartTokenizer = MBartTokenizer.from_pretrained(
            cls.checkpoint_name, src_lang="en_XX", tgt_lang="ro_RO"
        )
        cls.pad_token_id = 1
        return cls

    def check_language_codes(self):
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["ar_AR"], 250001)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["en_EN"], 250004)
        self.assertEqual(self.tokenizer.fairseq_tokens_to_ids["ro_RO"], 250020)

    def test_enro_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(RO_CODE, self.tokenizer.all_special_ids)
        generated_ids = [RO_CODE, 884, 9019, 96, 9, 916, 86792, 36, 18743, 15596, 5, 2]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_romanian = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_romanian)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_enro_tokenizer_truncation(self):
        src_text = ["this is gunna be a long sentence " * 20]
        assert isinstance(src_text[0], str)
        desired_max_length = 10
        ids = self.tokenizer(src_text, max_length=desired_max_length, truncation=True).input_ids[0]
        self.assertEqual(ids[-2], 2)
        self.assertEqual(ids[-1], EN_CODE)
        self.assertEqual(len(ids), desired_max_length)

    def test_mask_token(self):
        self.assertListEqual(self.tokenizer.convert_tokens_to_ids(["<mask>", "ar_AR"]), [250026, 250001])

    def test_special_tokens_unaffacted_by_save_load(self):
        tmpdirname = tempfile.mkdtemp()
        original_special_tokens = self.tokenizer.fairseq_tokens_to_ids
        self.tokenizer.save_pretrained(tmpdirname)
        new_tok = MBartTokenizer.from_pretrained(tmpdirname)
        self.assertDictEqual(new_tok.fairseq_tokens_to_ids, original_special_tokens)

    def test_seq2seq_max_length(self):
        batch = self.tokenizer(self.src_text, padding=True, truncation=True, max_length=3, return_tensors="pd")
        targets = self.tokenizer(self.tgt_text, padding=True, truncation=True, max_length=10, return_tensors="pd")
        labels = targets["input_ids"]
        batch["decoder_input_ids"] = shift_tokens_right(labels, self.tokenizer.pad_token_id)

        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(batch.decoder_input_ids.shape[1], 10)
