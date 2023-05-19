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

from paddlenlp.transformers.ernie_gram.tokenizer import ErnieGramTokenizer

from ..test_tokenizer_common import TokenizerTesterMixin, filter_non_english


class ErnieGramTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = ErnieGramTokenizer
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english
    test_seq2seq = True

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

        self.vocab_file = os.path.join(self.tmpdirname, ErnieGramTokenizer.resource_files_names["vocab_file"])

        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00E9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize("UNwant\u00E9d,running")
        self.assertListEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [9, 6, 7, 12, 10, 11])

    def test_offsets_mapping(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                text = "这世界很美"
                pair = "我们需要共同守护"

                # No pair
                tokens_with_offsets = tokenizer.encode(
                    text, return_special_tokens_mask=True, return_offsets_mapping=True, add_special_tokens=True
                )
                added_tokens = tokenizer.num_special_tokens_to_add(False)
                offsets = tokens_with_offsets["offset_mapping"]

                # Assert there is the same number of tokens and offsets
                self.assertEqual(len(offsets), len(tokens_with_offsets["input_ids"]))

                # Assert there is online added_tokens special_tokens
                self.assertEqual(sum(tokens_with_offsets["special_tokens_mask"]), added_tokens)

                # Pairs
                tokens_with_offsets = tokenizer.encode(
                    text, pair, return_special_tokens_mask=True, return_offsets_mapping=True, add_special_tokens=True
                )
                added_tokens = tokenizer.num_special_tokens_to_add(True)
                offsets = tokens_with_offsets["offset_mapping"]

                # Assert there is the same number of tokens and offsets
                self.assertEqual(len(offsets), len(tokens_with_offsets["input_ids"]))

                # Assert there is online added_tokens special_tokens
                self.assertEqual(sum(tokens_with_offsets["special_tokens_mask"]), added_tokens)

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual([tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]], [["[UNK]"], [], ["[UNK]"]])

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("ernie-gram-zh")

        text = tokenizer.encode("sequence builders", return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build", return_token_type_ids=None, add_special_tokens=False)[
            "input_ids"
        ]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [
            tokenizer.sep_token_id
        ]

    def test_offsets_with_special_characters(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                sentence = f"中国的首都 {tokenizer.mask_token} 是北京"
                tokens = tokenizer.encode(
                    sentence,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

                expected_results = [
                    ((0, 0), tokenizer.cls_token),
                    ((0, 1), "中"),
                    ((1, 2), "国"),
                    ((2, 3), "的"),
                    ((3, 4), "首"),
                    ((4, 5), "都"),
                    ((6, 12), "[MASK]"),
                    ((13, 14), "是"),
                    ((14, 15), "北"),
                    ((15, 16), "京"),
                    ((0, 0), tokenizer.sep_token),
                ]
                self.assertEqual(
                    [e[1] for e in expected_results], tokenizer.convert_ids_to_tokens(tokens["input_ids"])
                )

                self.assertEqual([e[0] for e in expected_results], tokens["offset_mapping"])

    def test_change_tokenize_chinese_chars(self):
        list_of_commun_chinese_char = ["的", "人", "有"]
        text_with_chinese_char = "".join(list_of_commun_chinese_char)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                kwargs["tokenize_chinese_chars"] = True
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer.encode(
                    text_with_chinese_char, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]

                tokens_without_spe_char_p = tokenizer.convert_ids_to_tokens(ids_without_spe_char_p)

                # it is expected that each Chinese character is not preceded by "##"
                self.assertListEqual(tokens_without_spe_char_p, list_of_commun_chinese_char)

                # not yet supported in bert tokenizer
                """
                kwargs["tokenize_chinese_chars"] = False
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer.encode(text_with_chinese_char, return_token_type_ids=None,add_special_tokens=False)["input_ids"]

                tokens_without_spe_char_p = tokenizer.convert_ids_to_tokens(ids_without_spe_char_p)

                # it is expected that only the first Chinese character is not preceded by "##".
                expected_tokens = [
                    f"##{token}" if idx != 0 else token for idx, token in enumerate(list_of_commun_chinese_char)
                ]
                self.assertListEqual(tokens_without_spe_char_p, expected_tokens)
                """
