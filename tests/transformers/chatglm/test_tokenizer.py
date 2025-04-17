# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from parameterized import parameterized

from paddlenlp.transformers import ChatGLMTokenizer
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

from ...transformers.test_tokenizer_common import TokenizerTesterMixin

VOCAB_FILES_NAMES = {
    "vocab_file": "ice_text.model",
}


class ChatGLMTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = ChatGLMTokenizer
    from_pretrained_vocab_key = "model_file"
    test_decode_token = True

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b", **kwargs)
        return tokenizer

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        text = "lower newer"
        bpe_tokens = ["▁lower", "▁newer"]
        tokens = tokenizer.tokenize(text, add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [680, 10243, 0]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_pretokenized_inputs(self, *args, **kwargs):
        pass

    def test_tokenizers_common_ids_setters(self, *args, **kwargs):
        pass

    def test_mask_output(self):
        pass

    def test_offsets_mapping(self):
        pass

    def test_offsets_mapping_with_unk(self):
        pass

    def test_special_tokens_mask(self):
        pass

    def test_special_tokens_mask_input_pairs(self):
        pass

    def test_right_and_left_padding(self):
        pass

    def test_encode_decode_with_spaces(self):
        # TODO Fix decode in tokenizer.
        pass

    def test_add_special_tokens(self):
        pass

    def test_padding_to_multiple_of(self):
        pass

    def test_padding_side_in_kwargs(self):
        tokenizer = self.get_tokenizer(padding_side="left")
        self.assertEqual(tokenizer.padding_side, "left")

        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.padding_side, "left")

    def test_truncation_side_in_kwargs(self):
        tokenizer = self.get_tokenizer(truncation_side="left")
        self.assertEqual(tokenizer.truncation_side, "left")

        tokenizer = self.get_tokenizer(truncation_side="right")
        self.assertEqual(tokenizer.truncation_side, "right")

    def test_add_tokens(self):
        tokenizer = self.get_tokenizer()

        vocab_size = len(tokenizer)
        self.assertEqual(tokenizer.add_tokens(""), 0)
        self.assertEqual(tokenizer.add_tokens("testoken"), 1)
        self.assertEqual(tokenizer.add_tokens(["testoken1", "testtoken2"]), 2)
        self.assertEqual(len(tokenizer), vocab_size + 3)

        self.assertEqual(tokenizer.add_special_tokens({}), 0)
        self.assertRaises(AssertionError, tokenizer.add_special_tokens, {"additional_special_tokens": "<testtoken1>"})
        self.assertEqual(tokenizer.add_special_tokens({"additional_special_tokens": ["<testtoken2>"]}), 1)
        self.assertEqual(
            tokenizer.add_special_tokens({"additional_special_tokens": ["<testtoken3>", "<testtoken4>"]}), 2
        )
        self.assertIn("<testtoken3>", tokenizer.special_tokens_map["additional_special_tokens"])
        self.assertIsInstance(tokenizer.special_tokens_map["additional_special_tokens"], list)
        self.assertGreaterEqual(len(tokenizer.special_tokens_map["additional_special_tokens"]), 2)

        self.assertEqual(len(tokenizer), vocab_size + 6)

    def test_add_tokens_tokenizer(self):
        tokenizer = self.get_tokenizer()

        vocab_size = tokenizer.vocab_size
        all_size = len(tokenizer)

        self.assertNotEqual(vocab_size, 0)

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
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

    def test_consecutive_unk_string(self):
        tokenizer = self.get_tokenizer(add_bos_token=False)

        tokens = [tokenizer.unk_token for _ in range(2)]
        string = tokenizer.convert_tokens_to_string(tokens)
        encoding = tokenizer(
            text=string,
            add_special_tokens=False,
            runcation=True,
            return_offsets_mapping=True,
        )
        # TODO (wanghuijuan): Aligned with transformers, but 2 expected.
        self.assertEqual(len(encoding["input_ids"]), 3)
        self.assertEqual(len(encoding["offset_mapping"]), 3)

    def test_padding_if_pad_token_set_slow(self):
        tokenizer = self.get_tokenizer()

        # Simple input
        s = "This is a simple input"
        s2 = ["This is a simple input looooooooong", "This is a simple input"]
        p = ("This is a simple input", "This is a pair")

        pad_token_id = tokenizer.pad_token_id

        out_s = tokenizer(s, padding="max_length", max_length=30, return_tensors="np", return_attention_mask=True)
        out_s2 = tokenizer(s2, padding=True, truncate=True, return_tensors="np", return_attention_mask=True)
        out_p = tokenizer(*p, padding="max_length", max_length=60, return_tensors="np", return_attention_mask=True)

        # s
        # test single string max_length padding
        self.assertEqual(out_s["input_ids"].shape[-1], 30)
        self.assertTrue(pad_token_id in out_s["input_ids"])
        self.assertTrue(0 in out_s["attention_mask"][..., 0])

        # s2
        # test automatic padding
        self.assertEqual(out_s2["input_ids"].shape[-1], 11)
        # long slice doesn't have padding
        self.assertFalse(pad_token_id in out_s2["input_ids"][0])
        self.assertFalse(0 in out_s2["attention_mask"][0][..., 0])
        # short slice does have padding
        self.assertTrue(pad_token_id in out_s2["input_ids"][1])
        self.assertTrue(0 in out_s2["attention_mask"][1][..., 0])

        # p
        # test single pair max_length padding
        self.assertEqual(out_p["input_ids"].shape[-1], 60)
        self.assertTrue(pad_token_id in out_p["input_ids"])
        self.assertTrue(0 in out_p["attention_mask"][..., 0])

    def test_add_bos_token_slow(self):
        tokenizer = self.get_tokenizer()

        s = "This is a simple input"
        s2 = ["This is a simple input 1", "This is a simple input 2"]

        bos_token_id = tokenizer.bos_token_id

        out_s = tokenizer(s)
        out_s2 = tokenizer(s2)

        self.assertEqual(out_s.input_ids[-1], bos_token_id)
        self.assertTrue(all(o[-1] == bos_token_id for o in out_s2["input_ids"]))

    def test_pretrained_model_lists(self):
        # No max_model_input_sizes
        self.assertGreaterEqual(len(self.tokenizer_class.pretrained_resource_files_map), 1)
        self.assertGreaterEqual(len(list(self.tokenizer_class.pretrained_resource_files_map.values())[0]), 1)

    @parameterized.expand([(True,), (False,)])
    def test_encode_plus_with_padding(self, use_padding_as_call_kwarg: bool):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_size = 10
                padding_idx = tokenizer.pad_token_id
                token_type_padding_idx = tokenizer.pad_token_type_id

                encoded_sequence = tokenizer.encode(sequence, return_special_tokens_mask=True)
                input_ids = encoded_sequence["input_ids"]
                special_tokens_mask = encoded_sequence["special_tokens_mask"]
                sequence_length = len(input_ids)

                # Test right padding
                tokenizer_kwargs_right = {
                    "max_length": sequence_length + padding_size,
                    "padding": "max_length",
                    "return_special_tokens_mask": True,
                }

                if not use_padding_as_call_kwarg:
                    tokenizer.padding_side = "right"
                else:
                    tokenizer_kwargs_right["padding_side"] = "right"
                self.assertRaises(AssertionError, lambda: tokenizer.encode_plus(sequence, **tokenizer_kwargs_right))

                # Test left padding
                tokenizer_kwargs_left = {
                    "max_length": sequence_length + padding_size,
                    "padding": "max_length",
                    "return_special_tokens_mask": True,
                }

                if not use_padding_as_call_kwarg:
                    tokenizer.padding_side = "left"
                else:
                    tokenizer_kwargs_left["padding_side"] = "left"

                left_padded_sequence = tokenizer.encode_plus(sequence, **tokenizer_kwargs_left)
                left_padded_input_ids = left_padded_sequence["input_ids"]
                left_padded_special_tokens_mask = left_padded_sequence["special_tokens_mask"]
                left_padded_sequence_length = len(left_padded_input_ids)

                self.assertEqual(sequence_length + padding_size, left_padded_sequence_length)
                self.assertEqual([padding_idx] * padding_size + input_ids, left_padded_input_ids)
                self.assertEqual([1] * padding_size + special_tokens_mask, left_padded_special_tokens_mask)

                if "token_type_ids" in tokenizer.model_input_names:
                    token_type_ids = encoded_sequence["token_type_ids"]
                    left_padded_token_type_ids = left_padded_sequence["token_type_ids"]

                    self.assertEqual(
                        [token_type_padding_idx] * padding_size + token_type_ids, left_padded_token_type_ids
                    )

                if "attention_mask" in tokenizer.model_input_names and "attention_mask" in encoded_sequence:
                    attention_mask = encoded_sequence["attention_mask"]
                    left_padded_attention_mask = left_padded_sequence["attention_mask"]

                    self.assertEqual([0] * padding_size + attention_mask, left_padded_attention_mask)

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
                tokenizer.padding_side = "left"
                encoded_sequence = tokenizer.encode(sequence)["input_ids"]
                sequence_length = len(encoded_sequence)
                # FIXME: the next line should be padding(max_length) to avoid warning
                padded_sequence = tokenizer.encode(
                    sequence, max_length=sequence_length + padding_size, pad_to_max_seq_len=True
                )["input_ids"]
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertEqual([padding_idx] * padding_size + encoded_sequence, padded_sequence)

                # Check that nothing is done when a maximum length is not specified
                encoded_sequence = tokenizer.encode(sequence)["input_ids"]
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(sequence, pad_to_max_seq_len=True)["input_ids"]
                padded_sequence_left_length = len(padded_sequence_left)
                self.assertEqual(sequence_length, padded_sequence_left_length)
                self.assertEqual(encoded_sequence, padded_sequence_left)

    def test_padding_with_attention_mask(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.pad_token is None:
                    self.skipTest("No padding token.")
                if "attention_mask" not in tokenizer.model_input_names:
                    self.skipTest("This model does not use attention mask.")

                features = [
                    {"input_ids": [1, 2, 3], "attention_mask": np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 1]]])},
                    {
                        "input_ids": [
                            1,
                            2,
                        ],
                        "attention_mask": np.array([[[0, 0], [0, 1]]]),
                    },
                ]
                padded_features = tokenizer.pad(features)
                print(padded_features["attention_mask"])
                self.assertListEqual(
                    [x.tolist() for x in padded_features["attention_mask"]],
                    [
                        [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                        [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                    ],
                )

    def test_batch_encode_plus_padding(self):
        # Test that padded sequences are equivalent between batch_encode_plus and encode_plus
        # Left padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokenizer.padding_side = "left"
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences = [
                    tokenizer.encode(sequence, max_length=max_length, padding="max_length") for sequence in sequences
                ]
                encoded_sequences_batch = tokenizer.batch_encode(
                    sequences, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    [x["input_ids"] for x in encoded_sequences],
                    [
                        x["input_ids"]
                        for x in self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                    ],
                )

    def test_batch_encode_plus_batch_sequence_length(self):
        # Tests that all encoded values have the correct size
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                def tolist(input_dict_list):
                    if isinstance(input_dict_list, np.ndarray):
                        return input_dict_list.tolist()
                    unwrap = False
                    if isinstance(input_dict_list, dict):
                        input_dict_list = [input_dict_list]
                        unwrap = True
                    for i, input_dict in enumerate(input_dict_list):
                        for k in input_dict:
                            if isinstance(input_dict[k], np.ndarray):
                                input_dict_list[i][k] = input_dict[k].tolist()
                    return input_dict_list[0] if unwrap else input_dict_list

                encoded_sequences = [tokenizer.encode(sequence) for sequence in sequences]
                encoded_sequences_batch = tokenizer.batch_encode(sequences, padding=False)
                self.assertListEqual(
                    tolist(encoded_sequences),
                    tolist(self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)),
                )

                maximum_length = len(
                    max([encoded_sequence["input_ids"] for encoded_sequence in encoded_sequences], key=len)
                )

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences_padded = [
                    tokenizer.encode(sequence, max_length=maximum_length, padding="max_length")
                    for sequence in sequences
                ]

                encoded_sequences_batch_padded = tokenizer.batch_encode(sequences, padding=True)
                self.assertListEqual(
                    tolist(encoded_sequences_padded),
                    tolist(self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch_padded)),
                )

                # check 'longest' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode(sequences, padding=True)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode(
                    sequences, max_length=maximum_length + 10, padding="longest"
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        [x.tolist() for x in encoded_sequences_batch_padded_1[key]]
                        if key != "input_ids"
                        else encoded_sequences_batch_padded_1[key],
                        [x.tolist() for x in encoded_sequences_batch_padded_2[key]]
                        if key != "input_ids"
                        else encoded_sequences_batch_padded_2[key],
                    )

                # check 'no_padding' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode(sequences, padding=False)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode(
                    sequences, max_length=maximum_length + 10, padding=False
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        [x.tolist() for x in encoded_sequences_batch_padded_1[key]]
                        if key != "input_ids"
                        else encoded_sequences_batch_padded_1[key],
                        [x.tolist() for x in encoded_sequences_batch_padded_2[key]]
                        if key != "input_ids"
                        else encoded_sequences_batch_padded_2[key],
                    )
