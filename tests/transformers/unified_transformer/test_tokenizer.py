# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import tempfile
import unittest
from typing import List

from paddlenlp.transformers import PretrainedTokenizer, UnifiedTransformerTokenizer
from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase

from ...testing_utils import get_tests_dir

SAMPLE_SENTENCEPIECE = get_tests_dir("fixtures/test_sentencepiece.model")
SAMPLE_VOCAB = get_tests_dir("fixtures/vocab.zh.txt")


class UnifiedTransformerTokenizationTest(unittest.TestCase):

    tokenizer_class = UnifiedTransformerTokenizer
    test_sentencepiece = True
    from_pretrained_vocab_key = "sentencepiece_model_file"
    test_seq2seq = False
    test_offsets = False

    space_between_special_tokens = False
    from_pretrained_kwargs = None
    from_pretrained_filter = None

    test_sentencepiece_ignore_case = False

    def setUp(self):
        super().setUp()

        tokenizers_list = [
            (
                self.tokenizer_class,
                pretrained_name,
                self.from_pretrained_kwargs if self.from_pretrained_kwargs is not None else {},
            )
            for pretrained_name in self.tokenizer_class.pretrained_resource_files_map[
                self.from_pretrained_vocab_key
            ].keys()
            if self.from_pretrained_filter is None
            or (self.from_pretrained_filter is not None and self.from_pretrained_filter(pretrained_name))
        ]
        self.tokenizers_list = tokenizers_list[:1]

        with open(f"{get_tests_dir()}/sample_text.txt", encoding="utf-8") as f_data:
            self._data = f_data.read().replace("\n\n", "\n").strip()

        self.tmpdirname = tempfile.mkdtemp()

        tokenizer = UnifiedTransformerTokenizer(SAMPLE_VOCAB, SAMPLE_SENTENCEPIECE)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizers(self, **kwargs) -> List[PretrainedTokenizerBase]:
        return [self.get_tokenizer(**kwargs)]

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def test_get_vocab(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_dict = tokenizer.get_vocab()
                self.assertIsInstance(vocab_dict, dict)
                self.assertGreaterEqual(len(tokenizer), len(vocab_dict))

                vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
                self.assertEqual(len(vocab), len(tokenizer))

                tokenizer.add_tokens(["asdfasdfasdfasdf"])
                vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
                self.assertEqual(len(vocab), len(tokenizer))

    def test_right_and_left_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(sequence)["input_ids"]
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    sequence, max_length=sequence_length + padding_size, padding="max_length"
                )["input_ids"]
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertEqual(encoded_sequence + [padding_idx] * padding_size, padded_sequence)

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "left"
                encoded_sequence = tokenizer.encode(sequence)["input_ids"]
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    sequence, max_length=sequence_length + padding_size, padding="max_length"
                )["input_ids"]
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertEqual([padding_idx] * padding_size + encoded_sequence, padded_sequence)

                # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_padding'
                encoded_sequence = tokenizer.encode(sequence)["input_ids"]
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(sequence, padding=True)["input_ids"]
                padded_sequence_right_length = len(padded_sequence_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertEqual(encoded_sequence, padded_sequence_right)

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(sequence, padding="longest")["input_ids"]
                padded_sequence_left_length = len(padded_sequence_left)
                self.assertEqual(sequence_length, padded_sequence_left_length)
                self.assertEqual(encoded_sequence, padded_sequence_left)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(sequence)["input_ids"]
                padded_sequence_right_length = len(padded_sequence_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertEqual(encoded_sequence, padded_sequence_right)

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(sequence, padding=False)["input_ids"]
                padded_sequence_left_length = len(padded_sequence_left)
                self.assertEqual(sequence_length, padded_sequence_left_length)
                self.assertEqual(encoded_sequence, padded_sequence_left)

    def test_right_and_left_truncation(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "This is a test sequence"

                # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                truncation_size = 3
                tokenizer.truncation_side = "right"
                encoded_sequence = tokenizer.encode(sequence, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]
                sequence_length = len(encoded_sequence)
                # Remove EOS/BOS tokens
                truncated_sequence = tokenizer.encode(
                    sequence,
                    max_length=sequence_length - truncation_size,
                    truncation=True,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                )["input_ids"]
                truncated_sequence_length = len(truncated_sequence)
                self.assertEqual(sequence_length, truncated_sequence_length + truncation_size)
                self.assertEqual(encoded_sequence[:-truncation_size], truncated_sequence)

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the truncation flag set to True
                tokenizer.truncation_side = "left"
                sequence_length = len(encoded_sequence)
                truncated_sequence = tokenizer.encode(
                    sequence,
                    max_length=sequence_length - truncation_size,
                    truncation=True,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                )["input_ids"]
                truncated_sequence_length = len(truncated_sequence)
                self.assertEqual(sequence_length, truncated_sequence_length + truncation_size)
                self.assertEqual(encoded_sequence[truncation_size:], truncated_sequence)

                # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_truncation'
                sequence_length = len(encoded_sequence)

                tokenizer.truncation_side = "right"
                truncated_sequence_right = tokenizer.encode(
                    sequence, truncation=True, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                truncated_sequence_right_length = len(truncated_sequence_right)
                self.assertEqual(sequence_length, truncated_sequence_right_length)
                self.assertEqual(encoded_sequence, truncated_sequence_right)

                tokenizer.truncation_side = "left"
                truncated_sequence_left = tokenizer.encode(
                    sequence, truncation="longest_first", return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                truncated_sequence_left_length = len(truncated_sequence_left)
                self.assertEqual(sequence_length, truncated_sequence_left_length)
                self.assertEqual(encoded_sequence, truncated_sequence_left)

                tokenizer.truncation_side = "right"
                truncated_sequence_right = tokenizer.encode(
                    sequence, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                truncated_sequence_right_length = len(truncated_sequence_right)
                self.assertEqual(sequence_length, truncated_sequence_right_length)
                self.assertEqual(encoded_sequence, truncated_sequence_right)

                tokenizer.truncation_side = "left"
                truncated_sequence_left = tokenizer.encode(
                    sequence, truncation=False, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                truncated_sequence_left_length = len(truncated_sequence_left)
                self.assertEqual(sequence_length, truncated_sequence_left_length)
                self.assertEqual(encoded_sequence, truncated_sequence_left)

    def test_padding_to_max_length(self):
        """We keep this test for backward compatibility but it should be remove when `pad_to_max_seq_len` is deprecated."""
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(sequence)["input_ids"]
                sequence_length = len(encoded_sequence)
                # FIXME: the next line should be padding(max_length) to avoid warning
                padded_sequence = tokenizer.encode(
                    sequence, max_length=sequence_length + padding_size, pad_to_max_seq_len=True
                )["input_ids"]
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertEqual(encoded_sequence + [padding_idx] * padding_size, padded_sequence)

                # Check that nothing is done when a maximum length is not specified
                encoded_sequence = tokenizer.encode(sequence)["input_ids"]
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(sequence, pad_to_max_seq_len=True)["input_ids"]
                padded_sequence_right_length = len(padded_sequence_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertEqual(encoded_sequence, padded_sequence_right)

    def _check_no_pad_token_padding(self, tokenizer, sequences):
        # if tokenizer does not have pad_token_id, an error should be thrown
        if tokenizer.pad_token_id is None:
            with self.assertRaises(ValueError):
                if isinstance(sequences, list):
                    tokenizer.batch_encode(sequences, padding="longest")
                else:
                    tokenizer.encode(sequences, padding=True)

            # add pad_token_id to pass subsequent tests
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    def test_convert_tokens_to_string_format(self):
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokens = ["今天", "天气"]
                string = tokenizer.convert_tokens_to_string(tokens)

                self.assertIsInstance(string, str)
