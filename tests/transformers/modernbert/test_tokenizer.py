# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import ModernBertTokenizer
from paddlenlp.transformers.bert.tokenizer import (
    BasicTokenizer,
    WordpieceTokenizer,
    _is_control,
    _is_punctuation,
    _is_whitespace,
)

from ...testing_utils import slow
from ...transformers.test_tokenizer_common import (
    TokenizerTesterMixin,
    filter_non_english,
)


class ModernBertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = ModernBertTokenizer
    space_between_special_tokens = True
    from_pretraind_filter = filter_non_english
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
            "##nt",
            "##ing",
            "##s",
            "##ly",
            "low",
            "##er",
            "##est",
            "hi",
            "##gh",
            "##est",
            "app",
            "##le",
            "pie",
            "##s",
            "juice",
            "phone",
            "##s",
            "like",
            "but",
            "##ter",
            "fly",
            "##ing",
            "end",
            "##s",
        ]

        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00E9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = ModernBertTokenizer(self.vocab_file)

        tokens = tokenizer.tokenize("UNwant\u00E9d,running")
        self.assertListEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [1, 6, 8, 2, 3, 11])

    def test_chinese(self):
        tokenizer = ModernBertTokenizer(self.vocab_file)

        self.assertListEqual(tokenizer.tokenize("ah\u535A\u63A8zz"), ["[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]"])

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual([tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]], [["test"], ["[UNK]"], ["test"]])

    @slow
    def test_sequence_builders(self):
        tokenizer = ModernBertTokenizer.from_pretrained("modernbert-base")

        text = tokenizer.encode("sequence builders", return_token_type_ids=True, add_special_tokens=True)
        text_2 = tokenizer.encode("multi-sequence build", return_token_type_ids=True, add_special_tokens=True)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text["input_ids"])
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text["input_ids"], text_2["input_ids"])

        assert encoded_sentence == [101] + text["input_ids"] + [102]
        assert encoded_pair == [101] + text["input_ids"] + [102] + text_2["input_ids"] + [102]


if __name__ == "__main__":
    unittest.main()
