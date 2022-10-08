# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import re
import unittest

from paddlenlp.transformers import CodeGenTokenizer
from paddlenlp.transformers.codegen.tokenizer import VOCAB_FILES_NAMES
from ...testing_utils import slow

from ..test_tokenizer_common import TokenizerTesterMixin


class CodeGenTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = CodeGenTokenizer
    from_pretrained_kwargs = {"add_prefix_space": True}
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
            "<|endoftext|>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = [
            "#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""
        ]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname,
                                       VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname,
                                        VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return CodeGenTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = CodeGenTokenizer(self.vocab_file, self.merges_file,
                                     **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["\u0120low", "er", "\u0120", "n", "e", "w", "er"]
        tokens = tokenizer.tokenize(text, add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [14, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens),
                             input_bpe_tokens)

    def test_pretokenized_inputs(self, *args, **kwargs):
        # It's very difficult to mix/test pretokenization with byte-level
        # And get both CodeGen and Roberta to work at the same time (mostly an issue of adding a space before the string)
        pass

    def test_padding_if_pad_token_set_slow(self):
        tokenizer = CodeGenTokenizer.from_pretrained(self.tmpdirname,
                                                     pad_token="<pad>")

        # Simple input
        s = "This is a simple input"
        s2 = ["This is a simple input looooooooong", "This is a simple input"]
        p = ("This is a simple input", "This is a pair")
        p2 = [
            ("This is a simple input loooooong", "This is a simple input"),
            ("This is a simple pair loooooong", "This is a simple pair"),
        ]

        pad_token_id = tokenizer.pad_token_id

        out_s = tokenizer(s,
                          padding="max_length",
                          max_length=30,
                          return_tensors="np",
                          return_attention_mask=True)
        out_s2 = tokenizer(s2,
                           padding=True,
                           truncate=True,
                           return_tensors="np",
                           return_attention_mask=True)
        out_p = tokenizer(*p,
                          padding="max_length",
                          max_length=60,
                          return_tensors="np",
                          return_attention_mask=True)
        out_p2 = tokenizer(p2,
                           padding=True,
                           truncate=True,
                           return_tensors="np",
                           return_attention_mask=True)

        # s
        # test single string max_length padding
        self.assertEqual(out_s["input_ids"].shape[-1], 30)
        self.assertTrue(pad_token_id in out_s["input_ids"])
        self.assertTrue(0 in out_s["attention_mask"])

        # s2
        # test automatic padding
        self.assertEqual(out_s2["input_ids"].shape[-1], 33)
        # long slice doesn't have padding
        self.assertFalse(pad_token_id in out_s2["input_ids"][0])
        self.assertFalse(0 in out_s2["attention_mask"][0])
        # short slice does have padding
        self.assertTrue(pad_token_id in out_s2["input_ids"][1])
        self.assertTrue(0 in out_s2["attention_mask"][1])

        # p
        # test single pair max_length padding
        self.assertEqual(out_p["input_ids"].shape[-1], 60)
        self.assertTrue(pad_token_id in out_p["input_ids"])
        self.assertTrue(0 in out_p["attention_mask"])

        # p2
        # test automatic padding pair
        self.assertEqual(out_p2["input_ids"].shape[-1], 52)
        # long slice pair doesn't have padding
        self.assertFalse(pad_token_id in out_p2["input_ids"][0])
        self.assertFalse(0 in out_p2["attention_mask"][0])
        # short slice pair does have padding
        self.assertTrue(pad_token_id in out_p2["input_ids"][1])
        self.assertTrue(0 in out_p2["attention_mask"][1])

    def test_add_bos_token_slow(self):
        bos_token = "$$$"
        tokenizer = CodeGenTokenizer.from_pretrained(self.tmpdirname,
                                                     bos_token=bos_token,
                                                     add_bos_token=True)

        s = "This is a simple input"
        s2 = ["This is a simple input 1", "This is a simple input 2"]

        bos_token_id = tokenizer.bos_token_id

        out_s = tokenizer(s)
        out_s2 = tokenizer(s2)

        self.assertEqual(out_s.input_ids[0], bos_token_id)
        self.assertTrue(all(o[0] == bos_token_id for o in out_s2.input_ids))

        decode_s = tokenizer.decode(out_s.input_ids)
        decode_s2 = tokenizer.batch_decode(out_s2.input_ids)

        self.assertEqual(decode_s.split()[0], bos_token)
        self.assertTrue(all(d.split()[0] == bos_token for d in decode_s2))

    @slow
    def test_truncation(self):
        tokenizer = CodeGenTokenizer.from_pretrained(
            "Salesforce/codegen-350M-mono")

        text = "\nif len_a > len_b:\n    result = a\nelse:\n    result = b\n\n\n\n#"
        expected_trucated_text = "\nif len_a > len_b:      result = a\nelse:      result = b"

        input_ids = tokenizer.encode(text)["input_ids"]
        truncation_pattern = [
            "^#", re.escape("<|endoftext|>"), "^'''", '^"""', "\n\n\n"
        ]
        decoded_text = tokenizer.decode(
            input_ids, truncate_before_pattern=truncation_pattern)
        self.assertEqual(decoded_text, expected_trucated_text)

    # tokenizer has no padding token
    def test_padding_different_model_input_name(self):
        pass

    def test_pretrained_model_lists(self):
        # We should have at least one default checkpoint for each tokenizer
        # We should specify the max input length as well (used in some part to list the pretrained checkpoints)
        self.assertGreaterEqual(
            len(self.tokenizer_class.pretrained_resource_files_map), 1)
        self.assertEqual(
            len(
                list(
                    self.tokenizer_class.pretrained_resource_files_map.values())
                [0]),
            len(self.tokenizer_class.max_model_input_sizes),
        )

        weights_list = list(self.tokenizer_class.max_model_input_sizes.keys())
        weights_lists_2 = []
        for file_id, map_list in self.tokenizer_class.pretrained_resource_files_map.items(
        ):
            weights_lists_2.append(list(map_list.keys()))

        for weights_list_2 in weights_lists_2:
            self.assertListEqual(weights_list, weights_list_2)
