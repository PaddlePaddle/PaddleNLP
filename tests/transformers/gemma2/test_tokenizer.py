#
#
#

import os
import unittest

import paddle

from paddlenlp.transformers import Gemma2Tokenizer
from paddlenlp.transformers.gemma2.tokenizer import VOCAB_FILES_NAMES

from tests.testing_utils import slow
from tests.transformers.test_tokenizer_common import TokenizerTesterMixin


class Gemma2TokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Gemma2Tokenizer
    test_rust_tokenizer = False
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        vocab_tokens = ["<pad>", "<eos>", "<bos>", "a", "b", "c", "d", "e", "f", "g", "h"]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "wb") as fp:
            fp.write(b"dummy content")

        tokenizer = Gemma2Tokenizer(self.vocab_file)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = Gemma2Tokenizer(self.vocab_file)

        tokens = tokenizer.tokenize("abcd")
        self.assertListEqual(tokens, ["a", "b", "c", "d"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [3, 4, 5, 6],
        )

        tokens = tokenizer.tokenize("abcd")
        self.assertListEqual(tokens, ["a", "b", "c", "d"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [3, 4, 5, 6],
        )

    def test_special_tokens(self):
        tokenizer = Gemma2Tokenizer(self.vocab_file)
        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(["<pad>", "<eos>", "<bos>"]),
            [0, 1, 2],
        )

    def test_add_bos_token(self):
        tokenizer = Gemma2Tokenizer(self.vocab_file, add_bos_token=True)
        output = tokenizer("abcd")
        self.assertEqual(output["input_ids"][0], tokenizer.bos_token_id)

    def test_add_eos_token(self):
        tokenizer = Gemma2Tokenizer(self.vocab_file, add_eos_token=True)
        output = tokenizer("abcd")
        self.assertEqual(output["input_ids"][-1], tokenizer.eos_token_id)


if __name__ == "__main__":
    unittest.main()
