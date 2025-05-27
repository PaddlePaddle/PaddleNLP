# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import unittest

from paddlenlp.transformers.bert.tokenizer import BasicTokenizer, WordpieceTokenizer
from paddlenlp.transformers.roberta.tokenizer import (
    RobertaBPETokenizer,
    RobertaChineseTokenizer,
)
from paddlenlp.transformers.tokenizer_utils import AddedToken

from ...testing_utils import slow
from ...transformers.test_tokenizer_common import TokenizerTesterMixin

VOCAB_FILES_NAMES = RobertaBPETokenizer.resource_files_names


class RobertaBPETokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = RobertaBPETokenizer
    test_offsets = False
    from_pretrained_kwargs = {"cls_token": "<s>"}

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
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["l", "o", "w", "er", "\u0120", "n", "e", "w", "er"]
        tokens = tokenizer.tokenize(text)  # , add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [0, 1, 2, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def roberta_dict_integration_testing(self):
        tokenizer = self.get_tokenizer()

        self.assertListEqual(tokenizer.encode("Hello world!", add_special_tokens=False), [0, 31414, 232, 328, 2])
        self.assertListEqual(
            tokenizer.encode("Hello world! cécé herlolip 418", add_special_tokens=False),
            [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2],
        )

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("roberta-base")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)["input_ids"]

        encoded_text_from_decode = tokenizer.encode(
            "sequence builders", add_special_tokens=True, add_prefix_space=False
        )["input_ids"]
        encoded_pair_from_decode = tokenizer.encode(
            "sequence builders", "multi-sequence build", add_special_tokens=True, add_prefix_space=False
        )["input_ids"]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == encoded_text_from_decode
        assert encoded_pair == encoded_pair_from_decode

    def test_space_encoding(self):
        tokenizer = self.get_tokenizer()

        sequence = "Encode this sequence."
        space_encoding = tokenizer.byte_encoder[" ".encode("utf-8")[0]]

        # Testing encoder arguments
        encoded = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=False)["input_ids"]
        first_char = tokenizer.convert_ids_to_tokens(encoded[0])[0]
        self.assertNotEqual(first_char, space_encoding)

        encoded = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=True)["input_ids"]
        first_char = tokenizer.convert_ids_to_tokens(encoded[0])[0]
        self.assertEqual(first_char, space_encoding)

        tokenizer.add_special_tokens({"bos_token": "<s>"})
        encoded = tokenizer.encode(sequence, add_special_tokens=True)["input_ids"]
        first_char = tokenizer.convert_ids_to_tokens(encoded[1])[0]
        self.assertNotEqual(first_char, space_encoding)

        # Testing spaces after special tokens
        mask = "<mask>"
        tokenizer.add_special_tokens(
            {"mask_token": AddedToken(mask, lstrip=True, rstrip=False)}
        )  # mask token has a left space
        mask_ind = tokenizer.convert_tokens_to_ids(mask)

        sequence = "Encode <mask> sequence"
        sequence_nospace = "Encode <mask>sequence"

        encoded = tokenizer.encode(sequence)["input_ids"]
        mask_loc = encoded.index(mask_ind)
        first_char = tokenizer.convert_ids_to_tokens(encoded[mask_loc + 1])[0]
        self.assertEqual(first_char, space_encoding)

        encoded = tokenizer.encode(sequence_nospace)["input_ids"]
        mask_loc = encoded.index(mask_ind)
        first_char = tokenizer.convert_ids_to_tokens(encoded[mask_loc + 1])[0]
        self.assertNotEqual(first_char, space_encoding)

    def test_pretokenized_inputs(self):
        pass

    def test_embeded_special_tokens(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                sentence = "A, <mask> AllenNLP sentence."
                tokens_p = tokenizer_p.encode(
                    sentence, add_special_tokens=True, return_token_type_ids=True, return_attention_mask=True
                )

                # token_type_ids should put 0 everywhere
                self.assertEqual(sum(tokens_p["token_type_ids"]), 0)

                # attention_mask should put 1 everywhere, so sum over length should be 1
                self.assertEqual(sum(tokens_p["attention_mask"]) / len(tokens_p["attention_mask"]), 1)

                tokens_p_str = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])

                # Rust correctly handles the space before the mask while python doesnt
                self.assertSequenceEqual(tokens_p["input_ids"], [0, 250, 6, 50264, 3823, 487, 21992, 3645, 4, 2])

                self.assertSequenceEqual(
                    tokens_p_str, ["<s>", "A", ",", "<mask>", "ĠAllen", "N", "LP", "Ġsentence", ".", "</s>"]
                )

    def test_offsets_mapping_with_different_add_prefix_space_and_trim_space_arguments(self):
        # Test which aims to verify that the offsets are well adapted to the argument `add_prefix_space` and
        # `trim_offsets`
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                text_of_1_token = "hello"  # `hello` is a token in the vocabulary of `pretrained_name`
                text = f"{text_of_1_token} {text_of_1_token}"

                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, add_prefix_space=True)
                encoding = tokenizer(
                    text, return_offsets_mapping=True, add_special_tokens=False, return_token_type_ids=None
                )
                self.assertEqual(encoding.offset_mapping[0], (1, len(text_of_1_token) + 1))
                self.assertEqual(
                    encoding.offset_mapping[1],
                    (len(text_of_1_token) + 2, len(text_of_1_token) + 2 + len(text_of_1_token)),
                )

                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, add_prefix_space=False)
                encoding = tokenizer(
                    text, return_offsets_mapping=True, add_special_tokens=False, return_token_type_ids=None
                )
                self.assertEqual(encoding.offset_mapping[0], (0, len(text_of_1_token)))
                self.assertEqual(
                    encoding.offset_mapping[1],
                    (len(text_of_1_token) + 1, len(text_of_1_token) + 1 + len(text_of_1_token)),
                )


class RobertaChineseTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = RobertaChineseTokenizer
    space_between_special_tokens = True
    test_seq2seq = True

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
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
            "你",
            "我",
            "人",
            "天",
        ]

        self.vocab_file = os.path.join(self.tmpdirname, RobertaChineseTokenizer.resource_files_names["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00E9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize("UNwant\u00E9d,running")
        self.assertListEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [9, 6, 7, 12, 10, 11])

    def test_chinese(self):
        tokenizer = BasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("ah\u535A\u63A8zz"), ["ah", "\u535A", "\u63A8", "zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["hello", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hällo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["h\u00E9llo"])

    def test_basic_tokenizer_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_default(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["hallo", "!", "how", "are", "you", "?"]
        )
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "), ["HeLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=False)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HäLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_no_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=True)

        self.assertListEqual(
            tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "), ["HaLLo", "!", "how", "Are", "yoU", "?"]
        )

    def test_basic_tokenizer_respects_never_split_tokens(self):
        tokenizer = BasicTokenizer(do_lower_case=False, never_split=["[UNK]"])

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU? [UNK]"), ["HeLLo", "!", "how", "Are", "yoU", "?", "[UNK]"]
        )

    def test_wordpiece_tokenizer(self):
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(tokenizer.tokenize("unwanted running"), ["un", "##want", "##ed", "runn", "##ing"])

        self.assertListEqual(tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual([tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]], [["[UNK]"], [], ["[UNK]"]])

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("hfl/roberta-wwm-ext")

        text = tokenizer.encode("sequence builders", return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build", return_token_type_ids=None, add_special_tokens=False)[
            "input_ids"
        ]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [101] + text + [102]
        assert encoded_pair == [101] + text + [102] + text_2 + [102]

    def test_change_tokenize_chinese_chars(self):
        list_of_commun_chinese_char = ["的", "人", "有"]
        text_with_chinese_char = "".join(list_of_commun_chinese_char)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                kwargs["tokenize_chinese_chars"] = True
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer.encode(
                    text_with_chinese_char, return_token_type_ids=None, add_special_tokens=False
                )["input_ids"]

                tokens_without_spe_char_p = tokenizer.convert_ids_to_tokens(ids_without_spe_char_p)

                # it is expected that each Chinese character is not preceded by "##"
                self.assertListEqual(tokens_without_spe_char_p, list_of_commun_chinese_char)
