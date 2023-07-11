# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The Fairseq Authors, Microsoft Research, and the HuggingFace Inc. team. All rights reserved.
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

from paddlenlp.transformers import AddedToken, SpeechT5Tokenizer

from ...testing_utils import slow
from ..test_utils import get_tests_dir

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe_char.model")
from ..test_tokenizer_common import TokenizerTesterMixin

SPIECE_UNDERLINE = "▁"


class SpeechT5TokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = SpeechT5Tokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True

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

                encoded_sequences = [tokenizer.encode(sequence) for sequence in sequences]
                encoded_sequences_batch = tokenizer.batch_encode(sequences, padding=False)
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
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
                    encoded_sequences_padded,
                    self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch_padded),
                )

                # check 'longest' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode(sequences, padding=True)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode(
                    sequences, max_length=maximum_length + 10, padding="longest"
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

                # check 'no_padding' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode(sequences, padding=False)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode(
                    sequences, max_length=maximum_length + 10, padding=False
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

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
        self.assertEqual(len(encoding["input_ids"]), 2)
        self.assertEqual(len(encoding["offset_mapping"]), 2)

    def setUp(self):
        super().setUp()
        # We have a SentencePiece fixture for testing
        tokenizer = SpeechT5Tokenizer(SAMPLE_VOCAB)

        mask_token = AddedToken("<mask>", lstrip=True, rstrip=False)
        tokenizer.mask_token = mask_token
        tokenizer.add_special_tokens({"mask_token": mask_token})
        tokenizer.add_tokens(["<ctc_blank>"])

        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5):
        input_text, output_text = self.get_input_output_texts(tokenizer)
        ids = tokenizer.encode(output_text, add_special_tokens=False)
        text = tokenizer.decode(ids["input_ids"], clean_up_tokenization_spaces=False)
        return text, ids["input_ids"]

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<pad>"
        token_id = 1

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<s>")
        self.assertEqual(vocab_keys[1], "<pad>")
        self.assertEqual(vocab_keys[-4], "œ")
        self.assertEqual(vocab_keys[-2], "<mask>")
        self.assertEqual(vocab_keys[-1], "<ctc_blank>")
        self.assertEqual(len(vocab_keys), 81)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 79)

    def test_add_tokens_tokenizer(self):
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

                tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)[
                    "input_ids"
                ]

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-3], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l", add_special_tokens=False
                )["input_ids"]

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-3], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-3], tokens[-4])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-3], tokenizer.pad_token_id)

    def test_pickle_subword_regularization_tokenizer(self):
        pass

    def test_subword_regularization_tokenizer(self):
        pass

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()

        tokens = tokenizer.tokenize("This is a test")
        # fmt: off
        self.assertListEqual(tokens, [SPIECE_UNDERLINE, 'T', 'h', 'i', 's', SPIECE_UNDERLINE, 'i', 's', SPIECE_UNDERLINE, 'a', SPIECE_UNDERLINE, 't', 'e', 's', 't'])
        # fmt: on

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [4, 32, 11, 10, 12, 4, 10, 12, 4, 7, 4, 6, 5, 12, 6],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            # fmt: off
            [SPIECE_UNDERLINE, 'I', SPIECE_UNDERLINE, 'w', 'a', 's', SPIECE_UNDERLINE, 'b', 'o', 'r', 'n', SPIECE_UNDERLINE, 'i', 'n', SPIECE_UNDERLINE, '92000', ',', SPIECE_UNDERLINE, 'a', 'n', 'd', SPIECE_UNDERLINE, 't', 'h', 'i', 's', SPIECE_UNDERLINE, 'i', 's', SPIECE_UNDERLINE, 'f', 'a', 'l', 's', 'é', '.']
            # fmt: on
        )

        ids = tokenizer.convert_tokens_to_ids(tokens)
        # fmt: off
        self.assertListEqual(ids, [4, 30, 4, 20, 7, 12, 4, 25, 8, 13, 9, 4, 10, 9, 4, 3, 23, 4, 7, 9, 14, 4, 6, 11, 10, 12, 4, 10, 12, 4, 19, 7, 15, 12, 73, 26])
        # fmt: on

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            # fmt: off
            [SPIECE_UNDERLINE, 'I', SPIECE_UNDERLINE, 'w', 'a', 's', SPIECE_UNDERLINE, 'b', 'o', 'r', 'n', SPIECE_UNDERLINE, 'i', 'n', SPIECE_UNDERLINE, '<unk>', ',', SPIECE_UNDERLINE, 'a', 'n', 'd', SPIECE_UNDERLINE, 't', 'h', 'i', 's', SPIECE_UNDERLINE, 'i', 's', SPIECE_UNDERLINE, 'f', 'a', 'l', 's', 'é', '.']
            # fmt: on
        )

    @slow
    def test_tokenizer_integration(self):
        # Use custom sequence because this tokenizer does not handle numbers.
        sequences = [
            "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides "
            "general-purpose architectures (BERT, GPT, RoBERTa, XLM, DistilBert, XLNet...) for Natural "
            "Language Understanding (NLU) and Natural Language Generation (NLG) with over thirty-two pretrained "
            "models in one hundred plus languages and deep interoperability between Jax, PyTorch and TensorFlow.",
            "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context in all layers.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # fmt: off
        expected_encoding = {
            'input_ids': [
                [4, 32, 13, 7, 9, 12, 19, 8, 13, 18, 5, 13, 12, 4, 64, 19, 8, 13, 18, 5, 13, 15, 22, 4, 28, 9, 8, 20, 9, 4, 7, 12, 4, 24, 22, 6, 8, 13, 17, 11, 39, 6, 13, 7, 9, 12, 19, 8, 13, 18, 5, 13, 12, 4, 7, 9, 14, 4, 24, 22, 6, 8, 13, 17, 11, 39, 24, 13, 5, 6, 13, 7, 10, 9, 5, 14, 39, 25, 5, 13, 6, 63, 4, 24, 13, 8, 27, 10, 14, 5, 12, 4, 21, 5, 9, 5, 13, 7, 15, 39, 24, 16, 13, 24, 8, 12, 5, 4, 7, 13, 17, 11, 10, 6, 5, 17, 6, 16, 13, 5, 12, 4, 64, 40, 47, 54, 32, 23, 4, 53, 49, 32, 23, 4, 54, 8, 40, 47, 54, 32, 7, 23, 4, 69, 52, 43, 23, 4, 51, 10, 12, 6, 10, 15, 40, 5, 13, 6, 23, 4, 69, 52, 48, 5, 6, 26, 26, 26, 63, 4, 19, 8, 13, 4, 48, 7, 6, 16, 13, 7, 15, 4, 52, 7, 9, 21, 16, 7, 21, 5, 4, 61, 9, 14, 5, 13, 12, 6, 7, 9, 14, 10, 9, 21, 4, 64, 48, 52, 61, 63, 4, 7, 9, 14, 4, 48, 7, 6, 16, 13, 7, 15, 4, 52, 7, 9, 21, 16, 7, 21, 5, 4, 53, 5, 9, 5, 13, 7, 6, 10, 8, 9, 4, 64, 48, 52, 53, 63, 4, 20, 10, 6, 11, 4, 8, 27, 5, 13, 4, 6, 11, 10, 13, 6, 22, 39, 6, 20, 8, 4, 24, 13, 5, 6, 13, 7, 10, 9, 5, 14, 4, 18, 8, 14, 5, 15, 12, 4, 10, 9, 4, 8, 9, 5, 4, 11, 16, 9, 14, 13, 5, 14, 4, 24, 15, 16, 12, 4, 15, 7, 9, 21, 16, 7, 21, 5, 12, 4, 7, 9, 14, 4, 14, 5, 5, 24, 4, 10, 9, 6, 5, 13, 8, 24, 5, 13, 7, 25, 10, 15, 10, 6, 22, 4, 25, 5, 6, 20, 5, 5, 9, 4, 58, 7, 37, 23, 4, 49, 22, 32, 8, 13, 17, 11, 4, 7, 9, 14, 4, 32, 5, 9, 12, 8, 13, 55, 15, 8, 20, 26, 2],
                [4, 40, 47, 54, 32, 4, 10, 12, 4, 14, 5, 12, 10, 21, 9, 5, 14, 4, 6, 8, 4, 24, 13, 5, 39, 6, 13, 7, 10, 9, 4, 14, 5, 5, 24, 4, 25, 10, 14, 10, 13, 5, 17, 6, 10, 8, 9, 7, 15, 4, 13, 5, 24, 13, 5, 12, 5, 9, 6, 7, 6, 10, 8, 9, 12, 4, 19, 13, 8, 18, 4, 16, 9, 15, 7, 25, 5, 15, 5, 14, 4, 6, 5, 37, 6, 4, 25, 22, 4, 46, 8, 10, 9, 6, 15, 22, 4, 17, 8, 9, 14, 10, 6, 10, 8, 9, 10, 9, 21, 4, 8, 9, 4, 25, 8, 6, 11, 4, 15, 5, 19, 6, 4, 7, 9, 14, 4, 13, 10, 21, 11, 6, 4, 17, 8, 9, 6, 5, 37, 6, 4, 10, 9, 4, 7, 15, 15, 4, 15, 7, 22, 5, 13, 12, 26, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [4, 32, 11, 5, 4, 45, 16, 10, 17, 28, 4, 25, 13, 8, 20, 9, 4, 19, 8, 37, 4, 46, 16, 18, 24, 12, 4, 8, 27, 5, 13, 4, 6, 11, 5, 4, 15, 7, 57, 22, 4, 14, 8, 21, 26, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            'attention_mask': [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        }
        # fmt: on

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="microsoft/speecht5_asr",
            revision="c5ef64c71905caeccde0e4462ef3f9077224c524",
            sequences=sequences,
        )
