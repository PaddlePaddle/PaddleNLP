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

from paddlenlp.transformers.ernie_doc.tokenizer import ErnieDocTokenizer

from ...testing_utils import slow
from ...transformers.test_tokenizer_common import (
    TokenizerTesterMixin,
    filter_non_english,
)


class ErnieTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = ErnieDocTokenizer
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

        self.vocab_file = os.path.join(self.tmpdirname, ErnieDocTokenizer.resource_files_names["vocab_file"])
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

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("ernie-1.0")

        text = tokenizer.encode("sequence builders", return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build", return_token_type_ids=None, add_special_tokens=False)[
            "input_ids"
        ]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
        assert encoded_sentence == [1] + text + [2]
        assert encoded_pair == [1] + text + [2] + text_2 + [2]

    def test_offsets_with_special_characters(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                sentence = f"A, naïve {tokenizer.mask_token} AllenNLP sentence."
                tokens = tokenizer.encode(
                    sentence,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

                do_lower_case = tokenizer.do_lower_case if hasattr(tokenizer, "do_lower_case") else False
                expected_results = (
                    [
                        ((0, 0), tokenizer.cls_token),
                        ((0, 1), "A"),
                        ((1, 2), ","),
                        ((3, 5), "na"),
                        ((5, 6), "##ï"),
                        ((6, 8), "##ve"),
                        ((9, 15), tokenizer.mask_token),
                        ((16, 21), "Allen"),
                        ((21, 23), "##NL"),
                        ((23, 24), "##P"),
                        ((25, 33), "sentence"),
                        ((33, 34), "."),
                        ((0, 0), tokenizer.sep_token),
                    ]
                    if not do_lower_case
                    else [
                        ((0, 0), tokenizer.cls_token),
                        ((0, 1), "a"),
                        ((1, 2), ","),
                        ((3, 5), "na"),
                        ((5, 8), "##ive"),
                        ((9, 15), tokenizer.mask_token),
                        ((16, 21), "allen"),
                        ((21, 22), "##n"),
                        ((22, 24), "##lp"),
                        ((25, 27), "se"),
                        ((27, 29), "##nt"),
                        ((29, 33), "##ence"),
                        ((33, 34), "."),
                        ((0, 0), tokenizer.sep_token),
                    ]
                )

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

    @slow
    def test_with_emoji(self):
        tokenizer = self.tokenizer_class.from_pretrained("ernie-1.0")
        text = "好👌🏻"
        prompt = "评价维度"
        encoding = tokenizer(
            text=text,
            text_pair=prompt,
            runcation=True,
            pad_to_max_seq_len=True,
            return_attention_mask=True,
            return_position_ids=True,
            return_offsets_mapping=True,
        )
        self.assertEqual(len(encoding["input_ids"]), 10)
        self.assertEqual(len(encoding["offset_mapping"]), 10)

        text = "好👌🏻👌🏻👌🏻👌🏻👌🏻👌🏻"
        prompt = "评价维度"
        encoding = tokenizer(
            text=text,
            text_pair=prompt,
            runcation=True,
            pad_to_max_seq_len=True,
            return_attention_mask=True,
            return_position_ids=True,
            return_offsets_mapping=True,
        )
        self.assertEqual(len(encoding["input_ids"]), 15)
        self.assertEqual(len(encoding["offset_mapping"]), 15)
