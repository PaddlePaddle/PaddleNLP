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
from typing import List
import shutil

import sentencepiece as spm
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizerBase, PretrainedTokenizer
from paddlenlp.transformers.ernie_m.tokenizer import ErnieMTokenizer
from paddlenlp.transformers.tokenizer_utils import _is_whitespace, _is_control, _is_punctuation

from tests.testing_utils import slow, get_tests_dir
from tests.transformers.test_tokenizer_common import TokenizerTesterMixin, filter_non_english

EN_SENTENCEPIECE = get_tests_dir("fixtures/sentencepiece.en.bpe.model")
EMPTY_VOCAB = get_tests_dir("fixtures/en.bpe.vocab.txt")


class ErnieMEnglishTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = ErnieMTokenizer
    space_between_special_tokens = True
    test_seq2seq = True

    def setUp(self):
        super().setUp()

        tokenizer = ErnieMTokenizer(vocab_file=EMPTY_VOCAB,
                                    sentencepiece_model_file=EN_SENTENCEPIECE,
                                    unk_token="<unk>")
        tokenizer.save_pretrained(self.tmpdirname)

        # with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
        #     vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        # shutil.copyfile(
        #     SAMPLE_VOCAB,
        #     os.path.join(
        #         self.tmpdirname, ErnieMTokenizer.
        #         resource_files_names["sentencepiece_model_file"]))

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        return ErnieMTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwanted, running"
        output_text = "unwanted,running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()

        tokens = tokenizer.tokenize("UNwanted,running")
        self.assertListEqual(
            tokens,
            ['un', 'wa', 'nt', 'e', 'd', ',', 'r', 'un', 'n', 'i', 'n', 'g'])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens),
                             [9, 5, 8, 23, 35, 27, 38, 9, 13, 37, 13, 36])

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual(
            [tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]],
            [['t', 'e', 's', 't'], ['\xad'], ['t', 'e', 's', 't']])

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

    def test_maximum_encoding_length_pair_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False,
                                         model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Build a sequence from our model's vocabulary
                stride = 2
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)
                if len(ids) <= 2 + stride:
                    seq_0 = (seq_0 + " ") * (2 + stride)
                    ids = None

                seq0_tokens = tokenizer.encode(
                    seq_0, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                self.assertGreater(len(seq0_tokens), 2 + stride)

                seq_1 = "This is another sentence to be encoded."
                seq1_tokens = tokenizer.encode(
                    seq_1, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                if abs(len(seq0_tokens) - len(seq1_tokens)) <= 2:
                    seq1_tokens = seq1_tokens + seq1_tokens
                    seq_1 = tokenizer.decode(seq1_tokens,
                                             clean_up_tokenization_spaces=False)
                seq1_tokens = tokenizer.encode(
                    seq_1, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']

                self.assertGreater(len(seq1_tokens), 2 + stride)

                smallest = seq1_tokens if len(seq0_tokens) > len(
                    seq1_tokens) else seq0_tokens

                # We are not using the special tokens - a bit too hard to test all the tokenizers with this
                # TODO try this again later
                sequence = tokenizer.encode(
                    seq_0,
                    seq_1,
                    return_token_type_ids=None,
                    add_special_tokens=False)[
                        'input_ids']  # , add_prefix_space=False)

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_2 = seq_0 * model_max_length
                self.assertGreater(len(seq_2), model_max_length)

                sequence1 = tokenizer(seq_1,
                                      return_token_type_ids=None,
                                      add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                sequence2 = tokenizer(
                    seq_2,
                    seq_1,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                )
                total_length2 = len(sequence2["input_ids"])
                self.assertLess(
                    total_length1, model_max_length - 10,
                    "Issue with the testing sequence, please update it.")

                # (wj-Mcat): the default TruncationStrategy in ernie-m tokenizer is `longest_first`, so it will truncate the sentence to model_max_length
                self.assertEqual(
                    total_length2, model_max_length,
                    "Issue with the testing sequence, please update it.")

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token
                    and tokenizer.pad_token_id >= 0 else [False])
                for padding_state in padding_strategies:
                    with self.subTest(
                            f"{tokenizer.__class__.__name__} Padding: {padding_state}"
                    ):
                        for truncation_state in [
                                True, "longest_first", "only_first"
                        ]:
                            with self.subTest(
                                    f"{tokenizer.__class__.__name__} Truncation: {truncation_state}"
                            ):
                                output = tokenizer(seq_2,
                                                   seq_1,
                                                   padding=padding_state,
                                                   truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"]),
                                                 model_max_length)

                                output = tokenizer([seq_2], [seq_1],
                                                   padding=padding_state,
                                                   truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"][0]),
                                                 model_max_length)

                        # Simple
                        output = tokenizer(seq_1,
                                           seq_2,
                                           padding=padding_state,
                                           truncation="only_second")
                        self.assertEqual(len(output["input_ids"]),
                                         model_max_length)

                        output = tokenizer([seq_1], [seq_2],
                                           padding=padding_state,
                                           truncation="only_second")
                        self.assertEqual(len(output["input_ids"][0]),
                                         model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("PaddleNLP",
                                             level="WARNING") as cm:
                            output = tokenizer(seq_1,
                                               seq_2,
                                               padding=padding_state,
                                               truncation=False)
                            self.assertNotEqual(len(output["input_ids"]),
                                                model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(cm.records[0].message.startswith(
                            "Token indices sequence length is longer than the specified maximum sequence length for this model"
                        ))

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("PaddleNLP",
                                             level="WARNING") as cm:
                            output = tokenizer([seq_1], [seq_2],
                                               padding=padding_state,
                                               truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]),
                                                model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(cm.records[0].message.startswith(
                            "Token indices sequence length is longer than the specified maximum sequence length for this model"
                        ))

                truncated_first_sequence = tokenizer.encode(
                    seq_0, return_token_type_ids=None, add_special_tokens=False
                )['input_ids'][:-2] + tokenizer.encode(
                    seq_1, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                truncated_second_sequence = (tokenizer.encode(
                    seq_0, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids'] + tokenizer.encode(
                        seq_1,
                        return_token_type_ids=None,
                        add_special_tokens=False)['input_ids'][:-2])
                truncated_longest_sequence = (
                    truncated_first_sequence
                    if len(seq0_tokens) > len(seq1_tokens) else
                    truncated_second_sequence)

                overflow_first_sequence = tokenizer.encode(
                    seq_0, return_token_type_ids=None, add_special_tokens=False
                )['input_ids'][-(2 + stride):] + tokenizer.encode(
                    seq_1, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                overflow_second_sequence = (tokenizer.encode(
                    seq_0, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids'] + tokenizer.encode(
                        seq_1,
                        return_token_type_ids=None,
                        add_special_tokens=False)['input_ids'][-(2 + stride):])
                overflow_longest_sequence = (overflow_first_sequence if
                                             len(seq0_tokens) > len(seq1_tokens)
                                             else overflow_second_sequence)

                with self.assertRaises(ValueError) as context:
                    information = tokenizer(
                        seq_0,
                        seq_1,
                        max_length=len(sequence) - 2,
                        return_token_type_ids=None,
                        add_special_tokens=False,
                        stride=stride,
                        truncation_strategy="longest_first",
                        return_overflowing_tokens=True,
                        # add_prefix_space=False,
                    )

                self.assertTrue(context.exception.args[0].startswith(
                    "Not possible to return overflowing tokens for pair of sequences with the "
                    "`longest_first`. Please select another truncation strategy than `longest_first`, "
                    "for instance `only_second` or `only_first`."))

                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                # No overflowing tokens when using 'longest' in python tokenizers
                with self.assertRaises(ValueError) as context:
                    information = tokenizer(
                        seq_0,
                        seq_1,
                        max_length=len(sequence) - 2,
                        return_token_type_ids=None,
                        add_special_tokens=False,
                        stride=stride,
                        truncation_strategy=True,
                        return_overflowing_tokens=True,
                        # add_prefix_space=False,
                    )

                self.assertTrue(context.exception.args[0].startswith(
                    "Not possible to return overflowing tokens for pair of sequences with the "
                    "`longest_first`. Please select another truncation strategy than `longest_first`, "
                    "for instance `only_second` or `only_first`."))

                information_first_truncated = tokenizer(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                    stride=stride,
                    truncation_strategy="only_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                truncated_sequence = information_first_truncated["input_ids"]
                overflowing_tokens = information_first_truncated[
                    "overflowing_tokens"]

                self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                self.assertEqual(truncated_sequence, truncated_first_sequence)

                self.assertEqual(len(overflowing_tokens), 2 + stride)
                self.assertEqual(overflowing_tokens,
                                 seq0_tokens[-(2 + stride):])

                information_second_truncated = tokenizer(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                    stride=stride,
                    truncation_strategy="only_second",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                truncated_sequence = information_second_truncated["input_ids"]
                overflowing_tokens = information_second_truncated[
                    "overflowing_tokens"]

                self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                self.assertEqual(truncated_sequence, truncated_second_sequence)

                self.assertEqual(len(overflowing_tokens), 2 + stride)
                self.assertEqual(overflowing_tokens,
                                 seq1_tokens[-(2 + stride):])

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
