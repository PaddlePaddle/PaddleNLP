# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import Gemma2Tokenizer

from ...testing_utils import slow
from ..test_tokenizer_common import TokenizerTesterMixin


class Gemma2TokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Gemma2Tokenizer
    test_rust_tokenizer = False
    test_offsets = False
    test_sentencepiece = True

    def setUp(self):
        super().setUp()

        # Adapted from Hugging Face tests
        vocab_tokens = ["<s>", "</s>", "<unk>", "<pad>", "▁this", "▁is", "▁a", "▁test"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.model")

        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(vocab_tokens))

    def test_full_tokenizer(self):
        tokenizer = Gemma2Tokenizer(self.vocab_file)

        tokens = tokenizer.tokenize("This is a test")
        self.assertEqual(tokens, ["▁this", "▁is", "▁a", "▁test"])

        self.assertEqual(tokenizer.convert_tokens_to_string(tokens), "This is a test")

        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertEqual(ids, [4, 5, 6, 7])

        ids = tokenizer.encode("This is a test")
        self.assertEqual(ids, [0, 4, 5, 6, 7, 1])  # [BOS] this is a test [EOS]

        decoded_str = tokenizer.decode(ids)
        self.assertEqual(decoded_str, "This is a test")

    def test_special_tokens(self):
        tokenizer = Gemma2Tokenizer(self.vocab_file)

        self.assertEqual(tokenizer.bos_token, "<s>")
        self.assertEqual(tokenizer.eos_token, "</s>")
        self.assertEqual(tokenizer.unk_token, "<unk>")
        self.assertEqual(tokenizer.pad_token, "<pad>")

    def test_sequence_builders(self):
        tokenizer = Gemma2Tokenizer(self.vocab_file)

        text = tokenizer.encode("sequence builders")
        text_2 = tokenizer.encode("multi-sequence build")

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [0] + text + [1]
        assert encoded_pair == [0] + text + [1] + text_2 + [1]

    @slow
    def test_tokenizer_integration(self):
        # Adapted from Hugging Face tests
        tokenizer = Gemma2Tokenizer.from_pretrained("gemma2-2b")

        sequences = [
            "Paddle NLP is a powerful natural language processing toolkit.",
            "Gemma2 is a new language model from Google.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # Test encode_plus
        encoded_sequences_1 = [tokenizer.encode_plus(s) for s in sequences]

        # Test batch_encode_plus
        encoded_sequences_2 = tokenizer.batch_encode_plus(sequences)

        # Verify the outputs are the same
        for enc_1, enc_2 in zip(encoded_sequences_1, zip(encoded_sequences_2["input_ids"])):
            self.assertEqual(enc_1["input_ids"], list(enc_2))
