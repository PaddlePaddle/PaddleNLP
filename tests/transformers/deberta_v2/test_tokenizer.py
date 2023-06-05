# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import DebertaV2Tokenizer

from ...testing_utils import get_tests_dir
from ..test_tokenizer_common import TokenizerTesterMixin

SAMPLE_VOCAB = get_tests_dir("fixtures/spiece.model")


class DebertaV2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = DebertaV2Tokenizer
    from_pretrained_kwargs = {"add_prefix_space": True}
    test_seq2seq = False
    from_pretrained_vocab_key = "sentencepiece_model_file"

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB, unk_token="<unk>")
        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<pad>"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token), token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id), token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())
        self.assertEqual(vocab_keys[0], "<pad>")
        self.assertEqual(vocab_keys[1], "<unk>")
        self.assertEqual(vocab_keys[-1], "[PAD]")
        self.assertEqual(len(vocab_keys), 30_001)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 30_000)

    def test_do_lower_case(self):
        # fmt: off
        sequence = " \tHeLLo!how  \n Are yoU?  "
        tokens_target = ["▁hello", "!", "how", "▁are", "▁you", "?"]
        # fmt: on

        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB, do_lower_case=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False)["input_ids"])

        self.assertListEqual(tokens, tokens_target)

    @unittest.skip("There is an inconsistency between slow and fast tokenizer due to a bug in the fast one.")
    def test_sentencepiece_tokenize_and_convert_tokens_to_string(self):
        pass

    @unittest.skip("There is an inconsistency between slow and fast tokenizer due to a bug in the fast one.")
    def test_sentencepiece_tokenize_and_decode(self):
        pass

    def test_split_by_punct(self):
        # fmt: off
        sequence = "I was born in 92000, and this is falsé."
        tokens_target = ["▁", "<unk>", "▁was", "▁born", "▁in", "▁9", "2000", "▁", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "▁", ".", ]
        # fmt: on

        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB, split_by_punct=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False)["input_ids"])

        self.assertListEqual(tokens, tokens_target)

    def test_do_lower_case_split_by_punct(self):
        # fmt: off
        sequence = "I was born in 92000, and this is falsé."
        tokens_target = ["▁i", "▁was", "▁born", "▁in", "▁9", "2000", "▁", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "▁", ".", ]
        # fmt: on

        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB, do_lower_case=True, split_by_punct=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False)["input_ids"])
        self.assertListEqual(tokens, tokens_target)

    def test_do_lower_case_split_by_punct_false(self):
        # fmt: off
        sequence = "I was born in 92000, and this is falsé."
        tokens_target = ["▁i", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", ".", ]
        # fmt: on

        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB, do_lower_case=True, split_by_punct=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False)["input_ids"])

        self.assertListEqual(tokens, tokens_target)

    def test_do_lower_case_false_split_by_punct(self):
        # fmt: off
        sequence = "I was born in 92000, and this is falsé."
        tokens_target = ["▁", "<unk>", "▁was", "▁born", "▁in", "▁9", "2000", "▁", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "▁", ".", ]
        # fmt: on

        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB, do_lower_case=False, split_by_punct=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False)["input_ids"])

        self.assertListEqual(tokens, tokens_target)

    def test_do_lower_case_false_split_by_punct_false(self):
        # fmt: off
        sequence = " \tHeLLo!how  \n Are yoU?  "
        tokens_target = ["▁", "<unk>", "e", "<unk>", "o", "!", "how", "▁", "<unk>", "re", "▁yo", "<unk>", "?"]
        # fmt: on

        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB, do_lower_case=False, split_by_punct=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False)["input_ids"])

        self.assertListEqual(tokens, tokens_target)

    def test_full_tokenizer(self):
        sequence = "This is a test"
        ids_target = [13, 1, 4398, 25, 21, 1289]
        tokens_target = ["▁", "T", "his", "▁is", "▁a", "▁test"]
        back_tokens_target = ["▁", "<unk>", "his", "▁is", "▁a", "▁test"]

        tokenizer = DebertaV2Tokenizer(SAMPLE_VOCAB, keep_accents=True)

        ids = tokenizer.encode(sequence, add_special_tokens=False)["input_ids"]
        self.assertListEqual(ids, ids_target)
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, tokens_target)
        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, back_tokens_target)

        # fmt: off
        sequence = "I was born in 92000, and this is falsé."
        ids_target = [13, 1, 23, 386, 19, 561, 3050, 15, 17, 48, 25, 8256, 18, 1, 9]
        tokens_target = ["▁", "I", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "é", ".", ]
        back_tokens_target = ["▁", "<unk>", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", ".", ]
        # fmt: on

        ids = tokenizer.encode(sequence, add_special_tokens=False)["input_ids"]
        self.assertListEqual(ids, ids_target)
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, tokens_target)
        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, back_tokens_target)
