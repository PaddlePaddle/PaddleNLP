# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from paddlenlp.transformers.gemma.tokenizer import GemmaTokenizer
from paddlenlp.transformers.gemma.tokenizer_fast import GemmaTokenizerFast
from paddlenlp.transformers.tokenizer_utils import AddedToken, PretrainedTokenizer

from ..test_tokenizer_common import TokenizerTesterMixin

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


class GemmaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = GemmaTokenizer
    rust_tokenizer_class = GemmaTokenizerFast
    # skip test_create_token_type_ids cause transformers skip it
    test_rust_tokenizer = False
    test_decode_token = True

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2b", **kwargs)
        return tokenizer

    def get_rust_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        tokenizer = GemmaTokenizerFast.from_pretrained("google/gemma-2b", **kwargs)
        return tokenizer

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        text = "lower newer"
        bpe_tokens = ["lower", "▁newer"]
        tokens = tokenizer.tokenize(text, add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [15964, 36649, 3]
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

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                r_output = tokenizer_r.encode("Hey this is a <special> token")["input_ids"]

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)["input_ids"]

                self.assertTrue(special_token_id[0] in r_output)

    def test_fast_special_tokens(self):
        slow_tokenizer = self.get_tokenizer()
        fast_tokenizer = self.get_rust_tokenizer()
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)["input_ids"]
        assert slow == [2, 235280, 6453, 2121]

        fast_tokenizer.add_eos_token = False
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)["input_ids"]
        assert fast == [2, 235280, 6453, 2121]

        fast_tokenizer.add_eos_token = True
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)["input_ids"]
        assert fast == [2, 235280, 6453, 2121, 1]

        slow_tokenizer.add_eos_token = True
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)["input_ids"]
        assert slow == [2, 235280, 6453, 2121, 1]

        self.tokenizer_class.add_eos_token = False
        self.rust_tokenizer_class.add_eos_token = False

    def test_special_tokens_mask(self):
        pass

    def test_special_tokens_mask_input_pairs(self):
        pass

    def test_padding_side_in_kwargs(self):
        tokenizer = self.get_tokenizer(padding_side="left")
        self.assertEqual(tokenizer.padding_side, "left")

        tokenizer = self.get_tokenizer(padding_side="right")
        self.assertEqual(tokenizer.padding_side, "right")

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
            runcation=True,
            return_offsets_mapping=True,
        )
        self.assertEqual(len(encoding["input_ids"]), 2)
        self.assertEqual(len(encoding["offset_mapping"]), 2)

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
        self.assertTrue(0 in out_s["attention_mask"])

        # s2
        # test automatic padding
        self.assertEqual(out_s2["input_ids"].shape[-1], 9)
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

    def test_add_bos_token_slow(self):
        tokenizer = self.get_tokenizer()

        s = "This is a simple input"
        s2 = ["This is a simple input 1", "This is a simple input 2"]

        bos_token_id = tokenizer.bos_token_id

        out_s = tokenizer(s, add_special_tokens=True)
        out_s2 = tokenizer(s2, add_special_tokens=True)

        self.assertEqual(out_s.input_ids[0], bos_token_id)
        self.assertTrue(all(o[0] == bos_token_id for o in out_s2["input_ids"]))

    def test_pretrained_model_lists(self):
        # No max_model_input_sizes
        self.assertGreaterEqual(len(self.tokenizer_class.pretrained_resource_files_map), 1)
        self.assertGreaterEqual(len(list(self.tokenizer_class.pretrained_resource_files_map.values())[0]), 1)

    def test_add_special_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, ids = "A", [235280]

                special_token = "[SPECIAL_TOKEN]"

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token = tokenizer.encode(
                    special_token, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                self.assertEqual(len(encoded_special_token), 1)

                text = tokenizer.decode(ids + encoded_special_token, clean_up_tokenization_spaces=False)
                encoded = tokenizer.encode(text, return_token_type_ids=None, add_special_tokens=False)["input_ids"]

                input_encoded = tokenizer.encode(input_text, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]
                special_token_id = tokenizer.encode(
                    special_token, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]
                self.assertEqual(encoded, input_encoded + special_token_id)
                decoded = tokenizer.decode(encoded, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_extract_non_learnable_parts(self):
        models_with_templates = ["google/gemma-2b-it", "google/gemma-7b-it"]
        dummy_conversastions = [
            ["Q.", "A."],
            ["Q.A.", "A."],
            ["Q?", "A!"],
        ]
        decode_outputs = [
            ["<bos><start_of_turn>user\nQ.<end_of_turn>\n<start_of_turn>model\n", "A.<end_of_turn>\n"],
            ["<start_of_turn>user\nQ.A.<end_of_turn>\n<start_of_turn>model\n", "A.<end_of_turn>\n"],
            ["<start_of_turn>user\nQ?<end_of_turn>\n<start_of_turn>model\n", "A!<end_of_turn>\n"],
        ]
        context_data = {}
        context_data["is_training"] = True
        for model_id in models_with_templates:
            tokenizer = GemmaTokenizer.from_pretrained(model_id)
            if tokenizer.chat_template is None:
                continue
            conversation_result: list[tuple[list[int], list[int]]] = tokenizer.encode_chat_inputs(
                dummy_conversastions,
                context_data=context_data,
            )
            for idx, round in enumerate(conversation_result["conversations"]):
                self.assertEqual(tokenizer.decode(round[0]), decode_outputs[idx][0])
                self.assertEqual(tokenizer.decode(round[1]), decode_outputs[idx][1])
