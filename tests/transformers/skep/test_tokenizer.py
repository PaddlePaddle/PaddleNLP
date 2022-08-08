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
import json

from paddlenlp.transformers.skep.tokenizer import SkepTokenizer, BasicTokenizer, WordpieceTokenizer, BpeEncoder
from paddlenlp.transformers.tokenizer_utils import _is_whitespace, _is_control, _is_punctuation

from tests.testing_utils import slow
from tests.transformers.test_tokenizer_common import TokenizerTesterMixin, filter_non_english


class BPETokenizerTest(unittest.TestCase):

    def setUp(self):
        self.tokenizer = BpeEncoder()

    def test_simple_encode(self):
        pass


class SkepTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = SkepTokenizer
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english
    test_seq2seq = True

    use_bpe_encoder = False

    def setUp(self):
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
        self.vocab_file, self.bpe_json_file, self.bpe_vocab_file = None, None, None
        self.vocab_file = os.path.join(
            self.tmpdirname, SkepTokenizer.resource_files_names["vocab_file"])

        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        if self.use_bpe_encoder:
            self.bpe_json_file = os.path.join(
                self.tmpdirname,
                SkepTokenizer.resource_files_names["bpe_json_file"])
            with open(self.bpe_json_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {token: index
                     for index, token in enumerate(vocab_tokens)},
                    f,
                    ensure_ascii=False)

            self.bpe_vocab_file = os.path.join(
                self.tmpdirname,
                SkepTokenizer.resource_files_names["bpe_vocab_file"])
            merges = [
                "#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r",
                ""
            ]
            with open(self.bpe_vocab_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(merges))

    def get_tokenizer(self, **kwargs):
        return self.tokenizer_class(vocab_file=self.vocab_file,
                                    bpe_vocab_file=self.bpe_vocab_file,
                                    bpe_json_file=self.bpe_json_file,
                                    use_bpe_encoder=self.use_bpe_encoder,
                                    **kwargs)

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

    def test_offsets_mapping(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(
                    f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, **kwargs)

                text = "这世界很美"
                pair = "我们需要共同守护"

                # No pair
                tokens_with_offsets = tokenizer.encode(
                    text,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=True,
                    add_special_tokens=True)
                added_tokens = tokenizer.num_special_tokens_to_add(False)
                offsets = tokens_with_offsets["offset_mapping"]

                # Assert there is the same number of tokens and offsets
                self.assertEqual(len(offsets),
                                 len(tokens_with_offsets["input_ids"]))

                # Assert there is online added_tokens special_tokens
                self.assertEqual(
                    sum(tokens_with_offsets["special_tokens_mask"]),
                    added_tokens)

                # Pairs
                tokens_with_offsets = tokenizer.encode(
                    text,
                    pair,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=True,
                    add_special_tokens=True)
                added_tokens = tokenizer.num_special_tokens_to_add(True)
                offsets = tokens_with_offsets["offset_mapping"]

                # Assert there is the same number of tokens and offsets
                self.assertEqual(len(offsets),
                                 len(tokens_with_offsets["input_ids"]))

                # Assert there is online added_tokens special_tokens
                self.assertEqual(
                    sum(tokens_with_offsets["special_tokens_mask"]),
                    added_tokens)

    def test_is_whitespace(self):
        self.assertTrue(_is_whitespace(" "))
        self.assertTrue(_is_whitespace("\t"))
        self.assertTrue(_is_whitespace("\r"))
        self.assertTrue(_is_whitespace("\n"))
        self.assertTrue(_is_whitespace("\u00A0"))

        self.assertFalse(_is_whitespace("A"))
        self.assertFalse(_is_whitespace("-"))

    def test_is_control(self):
        self.assertTrue(_is_control("\u0005"))

        self.assertFalse(_is_control("A"))
        self.assertFalse(_is_control(" "))
        self.assertFalse(_is_control("\t"))
        self.assertFalse(_is_control("\r"))

    def test_is_punctuation(self):
        self.assertTrue(_is_punctuation("-"))
        self.assertTrue(_is_punctuation("$"))
        self.assertTrue(_is_punctuation("`"))
        self.assertTrue(_is_punctuation("."))

        self.assertFalse(_is_punctuation("A"))
        self.assertFalse(_is_punctuation(" "))

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual(
            [tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]],
            [["[UNK]"], [], ["[UNK]"]])

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained(
            "skep_ernie_1.0_large_ch")

        text = tokenizer.encode("sequence builders",
                                return_token_type_ids=None,
                                add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build",
                                  return_token_type_ids=None,
                                  add_special_tokens=False)["input_ids"]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id
                                    ] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [
            tokenizer.sep_token_id
        ] + text_2 + [tokenizer.sep_token_id]

    def test_offsets_with_special_characters(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(
                    f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, **kwargs)

                sentence = f"北京的首都 {tokenizer.mask_token} 是北京"
                tokens = tokenizer.encode(
                    sentence,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

                expected_results = [
                    ((0, 0), tokenizer.cls_token),
                    ((0, 1), "北"),
                    ((1, 2), "京"),
                    ((2, 3), "的"),
                    ((3, 4), "首"),
                    ((4, 5), "都"),
                    ((6, 12), "[MASK]"),
                    ((13, 14), "是"),
                    ((14, 15), "北"),
                    ((15, 16), "京"),
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

                # not yet supported in bert tokenizer
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
