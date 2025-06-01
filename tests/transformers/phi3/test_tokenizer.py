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

from paddlenlp.transformers import Phi3Tokenizer
from ..test_tokenizer_common import TokenizerTesterMixin, filter_non_english


class Phi3TokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Phi3Tokenizer
    test_sentencepiece = True
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, "vocab.json")
        self.merges_file = os.path.join(self.tmpdirname, "merges.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(str(vocab_tokens))
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return Phi3Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "</s>"
        token_id = 21

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "l")
        self.assertEqual(vocab_keys[1], "o")
        self.assertEqual(vocab_keys[-1], "<pad>")
        self.assertEqual(len(vocab_keys), 23)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 23)

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()

        tokens = tokenizer.tokenize("lower newer")
        self.assertListEqual(tokens, ["▁low", "er", "▁newer"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [14, 15, 17])

        tokens = tokenizer.tokenize("lower newer")
        self.assertListEqual(tokens, ["▁low", "er", "▁newer"])

        text = "lower newer"
        token_ids = tokenizer.encode(text, add_special_tokens=False)["input_ids"]
        self.assertListEqual(token_ids, [14, 15, 17])

        decoded_text = tokenizer.decode(token_ids)
        self.assertEqual(text, decoded_text)

    def test_sequence_builders(self):
        tokenizer = self.get_tokenizer()

        text = tokenizer.encode("sequence builders", add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)["input_ids"]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [1] + text + [2]
        assert encoded_pair == [1] + text + [2] + text_2 + [2]

    def test_special_tokens_mask_input_pairs(self):
        tokenizer = self.get_tokenizer()

        text = "Encode this."
        text_2 = "This one too please."
        encoded_text = tokenizer.encode(text, add_special_tokens=False)["input_ids"]
        encoded_text_2 = tokenizer.encode(text_2, add_special_tokens=False)["input_ids"]

        special_tokens_mask = tokenizer.get_special_tokens_mask(encoded_text, encoded_text_2)
        special_tokens_mask_2 = tokenizer.get_special_tokens_mask(encoded_text, encoded_text_2, already_has_special_tokens=True)

        self.assertEqual(special_tokens_mask, [1] + [0] * len(encoded_text) + [1] + [0] * len(encoded_text_2) + [1])
        self.assertEqual(special_tokens_mask_2, special_tokens_mask)


if __name__ == "__main__":
    unittest.main()
