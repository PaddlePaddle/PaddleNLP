# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Team. All rights reserved.
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


import json
import os
import unittest

from paddlenlp.transformers import Qwen2Tokenizer
from paddlenlp.transformers.qwen2.tokenizer import VOCAB_FILES_NAMES, bytes_to_unicode
from tests.transformers.test_tokenizer_common import TokenizerTesterMixin


class Qwen2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "__internal_testing__/tiny-random-qwen2"
    tokenizer_class = Qwen2Tokenizer
    test_slow_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        # this make sure the vocabuary is complete at the byte level.
        vocab = list(bytes_to_unicode().values())
        # the vocabulary, note:
        # - `"\u0120n"`, `"\u0120lowest"`, `"\u0120newer"`, and `"\u0120wider"` are ineffective, because there are
        #   not in the merges.
        # - `"01"` is ineffective, because the merge is ineffective due to pretokenization.
        vocab.extend(
            [
                "\u0120l",
                "\u0120n",
                "\u0120lo",
                "\u0120low",
                "er",
                "\u0120lowest",
                "\u0120newer",
                "\u0120wider",
                "01",
                ";}",
                ";}\u010a",
                "\u00cf\u0135",
                "\u0120#",
                "##",
            ]
        )

        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        # note: `"0 1"` is in the merges, but the pretokenization rules render it ineffective
        merges = [
            "#version: 0.2",
            "\u0120 l",
            "\u0120l o",
            "\u0120lo w",
            "e r",
            "0 1",
            "; }",
            ";} \u010a",
            "\u00cf \u0135",
            "\u0120 #",
            "# #",
        ]

        self.special_tokens_map = {"eos_token": "<|endoftext|>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return Qwen2Tokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        # this case should cover
        # - NFC normalization (code point U+03D3 has different normalization forms under NFC, NFD, NFKC, and NFKD)
        # - the pretokenization rules (spliting digits and merging symbols with \n\r)
        input_text = "lower lower newer 010;}\n<|endoftext|>\u03d2\u0301"
        output_text = "lower lower newer 010;}\n<|endoftext|>\u03d3"
        return input_text, output_text

    def test_python_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        sequence, _ = self.get_input_output_texts(tokenizer)
        bpe_tokens = [
            "l",
            "o",
            "w",
            "er",
            "\u0120low",
            "er",
            "\u0120",
            "n",
            "e",
            "w",
            "er",
            "\u0120",
            "0",
            "1",
            "0",
            ";}\u010a",
            "<|endoftext|>",
            "\u00cf\u0135",
        ]
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens
        input_bpe_tokens = [75, 78, 86, 260, 259, 260, 220, 77, 68, 86, 260, 220, 15, 16, 15, 266, 270, 267]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    @unittest.skip("We disable the test of pretokenization as it is not reversible.")
    def test_pretokenized_inputs(self):
        # the test case in parent class uses str.split to "pretokenize",
        # which eats the whitespaces, which, in turn, is not reversible.
        # the results, by nature, should be different.
        pass

    @unittest.skip("We disable the test of clean up tokenization spaces as it is not applicable.")
    def test_clean_up_tokenization_spaces(self):
        # it only tests bert-base-uncased and clean_up_tokenization_spaces is not applicable to this tokenizer
        pass

    def test_nfc_normalization(self):
        # per https://unicode.org/faq/normalization.html, there are three characters whose normalization forms
        # under NFC, NFD, NFKC, and NFKD are all different
        # using these, we can make sure only NFC is applied
        input_string = "\u03d2\u0301\u03d2\u0308\u017f\u0307"  # the NFD form
        output_string = "\u03d3\u03d4\u1e9b"  # the NFC form

        if self.test_slow_tokenizer:
            tokenizer = self.get_tokenizer()
            tokenizer_output_string, _ = tokenizer.prepare_for_tokenization(input_string)
            self.assertEqual(tokenizer_output_string, output_string)

    def test_slow_tokenizer_token_with_number_sign(self):
        if not self.test_slow_tokenizer:
            return

        sequence = " ###"
        token_ids = [268, 269]

        tokenizer = self.get_tokenizer()
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence)), token_ids)

    def test_tokenizer_integration(self):
        sequences = [
            "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides "
            "general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural "
            "Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained "
            "models in 100+ languages and deep interoperability between Jax, PyTorch and TensorFlow.",
            "🤗 Transformers 提供了可以轻松地下载并且训练先进的预训练模型的 API 和工具。使用预训练模型可以减少计算消耗和碳排放，并且节省从头训练所需要的时间和资源。",
            """```python\ntokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-tokenizer")\n"""
            """tokenizer("世界，你好！")```""",
        ]

        expected_encoding = {'input_ids': [[8963, 388, 320, 69514, 3881, 438, 4510, 27414, 32852, 388, 323, 4510, 27414, 21334, 35722, 1455, 529, 8, 5707, 4586, 58238, 77235, 320, 61437, 11, 479, 2828, 12, 17, 11, 11830, 61437, 64, 11, 1599, 10994, 11, 27604, 321, 33, 529, 11, 29881, 6954, 32574, 369, 18448, 11434, 45451, 320, 45, 23236, 8, 323, 18448, 11434, 23470, 320, 30042, 38, 8, 448, 916, 220, 18, 17, 10, 80669, 4119, 304, 220, 16, 15, 15, 10, 15459, 323, 5538, 94130, 2897, 1948, 619, 706, 11, 5355, 51, 21584, 323, 94986, 13], [144834, 80532, 93685, 83744, 34187, 73670, 104261, 29490, 62189, 103937, 104034, 102830, 98841, 104034, 104949, 9370, 5333, 58143, 102011, 1773, 37029, 98841, 104034, 104949, 73670, 101940, 100768, 104997, 33108, 100912, 105054, 90395, 100136, 106831, 45181, 64355, 104034, 113521, 101975, 33108, 85329, 1773, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643], [73594, 12669, 198, 85593, 284, 8979, 37434, 6387, 10442, 35722, 445, 48, 16948, 45274, 16948, 34841, 3135, 1138, 85593, 445, 99489, 3837, 108386, 6313, 899, 73594, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # fmt: skip

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="__internal_testing__/tiny-random-qwen2",
            sequences=sequences,
        )

    def test_consecutive_unk_string(self):
        tokenizer = self.get_tokenizer(add_bos_token=False)

        tokens = [tokenizer.unk_token for _ in range(2)]
        string = tokenizer.convert_tokens_to_string(tokens)
        encoding = tokenizer(
            text=string,
            truncation=True,
            return_offsets_mapping=True,
        )
        self.assertEqual(len(encoding["input_ids"]), 2)
        self.assertEqual(len(encoding["offset_mapping"]), 2)
