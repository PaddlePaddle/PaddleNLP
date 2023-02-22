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

from paddlenlp.transformers import MegatronBertTokenizer

from ...testing_utils import slow
from ...transformers.test_tokenizer_common import (
    TokenizerTesterMixin,
    filter_non_english,
)


class MegatronBertTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = MegatronBertTokenizer
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english
    test_seq2seq = False

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

        self.vocab_file = os.path.join(self.tmpdirname, MegatronBertTokenizer.resource_files_names["vocab_file"])
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

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()
        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual([tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]], [["[UNK]"], [], ["[UNK]"]])

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("megatronbert-uncased")

        text = tokenizer.encode("sequence builders", return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build", return_token_type_ids=None, add_special_tokens=False)[
            "input_ids"
        ]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [101] + text + [102]
        assert encoded_pair == [101] + text + [102] + text_2 + [102]

    def test_offsets_with_special_characters(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_fast = self.fast_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                sentence = f"A, naïve {tokenizer.mask_token} AllenNLP sentence."
                tokens = tokenizer.encode(
                    sentence,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

                tokens_fast = tokenizer_fast.encode(
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
                        ((3, 8), "naive"),
                        ((9, 15), tokenizer.mask_token),
                        ((16, 21), "allen"),
                        ((21, 23), "##nl"),
                        ((23, 24), "##p"),
                        ((25, 33), "sentence"),
                        ((33, 34), "."),
                        ((0, 0), tokenizer.sep_token),
                    ]
                )

                self.assertEqual(
                    [e[1] for e in expected_results], tokenizer.convert_ids_to_tokens(tokens["input_ids"])
                )
                self.assertEqual(
                    [e[1] for e in expected_results], tokenizer_fast.convert_ids_to_tokens(tokens_fast["input_ids"])
                )
                self.assertEqual([e[0] for e in expected_results], tokens["offset_mapping"])
                self.assertEqual([e[0] for e in expected_results], tokens_fast["offset_mapping"])

    def test_change_tokenize_chinese_chars(self):
        list_of_commun_chinese_char = ["的", "人", "有"]
        text_with_chinese_char = "".join(list_of_commun_chinese_char)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                kwargs["tokenize_chinese_chars"] = True
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_fast = self.fast_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer.encode(
                    text_with_chinese_char, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                ids_without_spe_char_fast = tokenizer_fast.encode(
                    text_with_chinese_char, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]

                tokens_without_spe_char_p = tokenizer.convert_ids_to_tokens(ids_without_spe_char_p)
                tokens_without_spe_char_fast = tokenizer_fast.convert_ids_to_tokens(ids_without_spe_char_fast)

                # it is expected that each Chinese character is not preceded by "##"
                self.assertListEqual(tokens_without_spe_char_p, list_of_commun_chinese_char)
                self.assertListEqual(tokens_without_spe_char_fast, list_of_commun_chinese_char)
