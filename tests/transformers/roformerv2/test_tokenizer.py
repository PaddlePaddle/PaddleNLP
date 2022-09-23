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
import unittest

from paddlenlp.data.vocab import Vocab

from paddlenlp.transformers.roformerv2.tokenizer import (BasicTokenizer,
                                                         RoFormerv2Tokenizer,
                                                         WordpieceTokenizer)

from tests.testing_utils import slow
from tests.transformers.test_tokenizer_common import TokenizerTesterMixin, filter_non_english


class RoFormerv2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = RoFormerv2Tokenizer
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english
    test_seq2seq = True

    def setUp(self):
        self.from_pretrained_kwargs = {"do_lower_case": False}

        super().setUp()
        vocab_tokens = [
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

        self.vocab_file = os.path.join(
            self.tmpdirname,
            RoFormerv2Tokenizer.resource_files_names["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        self.vocab = Vocab.from_dict(
            {token: index
             for index, token in enumerate(vocab_tokens)},
            unk_token='[UNK]',
            pad_token='[PAD]',
            bos_token='[CLS]',
            eos_token='[SEP]',
        )

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00E9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize("UNwant\u00E9d,running")
        self.assertListEqual(tokens,
                             ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens),
                             [9, 6, 7, 12, 10, 11])

    def test_chinese(self):
        tokenizer = BasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("ah\u535A\u63A8zz"),
                             ["ah", "\u535A", "\u63A8", "zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "),
                             ["hello", "!", "how", "are", "you", "?"])
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=False)

        self.assertListEqual(tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "),
                             ["hällo", "!", "how", "are", "you", "?"])
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["h\u00E9llo"])

    def test_basic_tokenizer_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "),
                             ["hallo", "!", "how", "are", "you", "?"])
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_default(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "),
                             ["hallo", "!", "how", "are", "you", "?"])
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=False)

        self.assertListEqual(tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "),
                             ["HeLLo", "!", "how", "Are", "yoU", "?"])

    def test_basic_tokenizer_no_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=False)

        self.assertListEqual(tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "),
                             ["HäLLo", "!", "how", "Are", "yoU", "?"])

    def test_basic_tokenizer_no_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=True)

        self.assertListEqual(tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "),
                             ["HaLLo", "!", "how", "Are", "yoU", "?"])

    def test_basic_tokenizer_respects_never_split_tokens(self):
        tokenizer = BasicTokenizer(do_lower_case=False, never_split=["[UNK]"])

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU? [UNK]"),
            ["HeLLo", "!", "how", "Are", "yoU", "?", "[UNK]"])

    def test_wordpiece_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un",
            "runn", "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(tokenizer.tokenize("unwanted running"),
                             ["un", "##want", "##ed", "runn", "##ing"])

        self.assertListEqual(tokenizer.tokenize("unwantedX running"),
                             ["[UNK]", "runn", "##ing"])

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual(
            [tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]],
            [["[UNK]"], [], ["[UNK]"]])

    # @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained(
            "roformer-chinese-small")

        text = tokenizer.encode("sequence builders",
                                return_token_type_ids=None,
                                add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build",
                                  return_token_type_ids=None,
                                  add_special_tokens=False)["input_ids"]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [101] + text + [102]
        assert encoded_pair == [101] + text + [102] + text_2 + [102]

    def test_offsets_with_special_characters(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(
                    f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, **kwargs)

                # sentence = f"testing with {tokenizer.mask_token} simple sentence"
                sentence = f"a simple {tokenizer.mask_token} allennlp sentence."
                tokens = tokenizer.encode(
                    sentence,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )
                expected_results = [
                    ((0, 0), tokenizer.cls_token),
                    ((0, 1), "a"),
                    ((2, 8), "simple"),
                    ((9, 15), tokenizer.mask_token),
                    ((16, 21), "allen"),
                    ((21, 23), "##nl"),
                    ((23, 24), "##p"),
                    ((25, 33), "sentence"),
                    ((33, 34), "."),
                    ((0, 0), tokenizer.sep_token),
                ]

                self.assertEqual([e[1] for e in expected_results],
                                 tokenizer.convert_ids_to_tokens(
                                     tokens["input_ids"]))
                self.assertEqual([e[0] for e in expected_results],
                                 tokens["offset_mapping"])

    def test_change_tokenize_chinese_chars(self):
        list_of_commun_chinese_char = ["的", "人", "有"]
        text_with_chinese_char = "".join(list_of_commun_chinese_char)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(
                    f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                kwargs["tokenize_chinese_chars"] = True
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer.encode(
                    text_with_chinese_char,
                    return_token_type_ids=None,
                    add_special_tokens=False)["input_ids"]

                tokens_without_spe_char_p = tokenizer.convert_ids_to_tokens(
                    ids_without_spe_char_p)

                # it is expected that each Chinese character is not preceded by "##"
                self.assertListEqual(tokens_without_spe_char_p,
                                     list_of_commun_chinese_char)
                '''
                kwargs["tokenize_chinese_chars"] = False
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer.encode(text_with_chinese_char, return_token_type_ids=None,add_special_tokens=False)["input_ids"]

                tokens_without_spe_char_p = tokenizer.convert_ids_to_tokens(ids_without_spe_char_p)

                # it is expected that only the first Chinese character is not preceded by "##".
                expected_tokens = [
                    f"##{token}" if idx != 0 else token for idx, token in enumerate(list_of_commun_chinese_char)
                ]
                self.assertListEqual(tokens_without_spe_char_p, expected_tokens)
                '''
