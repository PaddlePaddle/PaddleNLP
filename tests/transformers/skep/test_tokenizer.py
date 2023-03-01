# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import shutil
import unittest
from typing import Any, Dict, List, Tuple

from paddlenlp.transformers.skep.tokenizer import (
    BasicTokenizer,
    BpeEncoder,
    SkepTokenizer,
    WordpieceTokenizer,
)

from ...testing_utils import get_tests_dir, slow
from ..test_tokenizer_common import TokenizerTesterMixin, filter_non_english


def _class_name_func(cls, num: int, params_dict: Dict[str, Any]):
    suffix = "UseBPE" if params_dict["use_bpe_encoder"] else "NotUseBPE"
    return f"{cls.__name__}{suffix}"


def _read_tokens_from_file(file: str) -> List[str]:
    with open(file, "r", encoding="utf-8") as f:
        tokens = [token.strip() for token in f.readlines()]
    return tokens


class SkepBpeEncoderTest(unittest.TestCase):
    def setUp(self):
        self.vocab_file = get_tests_dir("fixtures/bpe.en/vocab.json")
        self.merges_file = get_tests_dir("fixtures/bpe.en/merges.txt")
        self.encoder = BpeEncoder(encoder_json_file=self.vocab_file, vocab_bpe_file=self.merges_file)

    def test_tokenizer(self):
        text = " lower newer"
        bpe_tokens = ["\u0120low", "er", "\u0120", "n", "e", "w", "er"]
        tokens = self.encoder._tokenize(text)

        self.assertListEqual(tokens, bpe_tokens)

        decoded_text = self.encoder.convert_tokens_to_string(tokens)
        self.assertEqual(text, decoded_text)

    def test_tokenizer_encode_decode(self):
        text = " lower newer"
        bpe_tokens = ["\u0120low", "er", "\u0120", "n", "e", "w", "er"]
        token_ids = self.encoder.encode(text)
        tokens = [self.encoder.decoder[token_id] for token_id in token_ids]

        self.assertListEqual(tokens, bpe_tokens)

        decoded_text = self.encoder.decode(token_ids)
        self.assertEqual(text, decoded_text)

    def test_unk_word(self):
        text = " lower newer a"
        with self.assertRaises(KeyError):
            self.encoder.encode(text)

        # can tokenize correct
        tokens = self.encoder._tokenize(text)

        # recognize the `a` as the <unk-token>
        token_ids = [self.encoder._convert_token_to_id(token) for token in tokens]

        decoded_tokens = [self.encoder._convert_id_to_token(token_id) for token_id in token_ids]
        self.assertIn(self.encoder.unk_token, decoded_tokens)


class SkepBPETokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = SkepTokenizer
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english
    test_seq2seq = True
    use_bpe_encoder = True

    def setUp(self):
        super().setUp()

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
        # save vocab file
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as f:
            # f.write('\n'.join(vocab))
            f.write("\n".join(vocab + ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]))

        # save bpe related files
        self.bpe_json_file = os.path.join(self.tmpdirname, "encoder.json")
        self.bpe_vocab_file = os.path.join(self.tmpdirname, "merges.txt")
        shutil.copyfile(get_tests_dir("fixtures/bpe.en/vocab.json"), self.bpe_json_file)

        shutil.copyfile(get_tests_dir("fixtures/bpe.en/merges.txt"), self.bpe_vocab_file)

    def get_tokenizer(self, **kwargs):
        tokenizer = self.tokenizer_class.from_pretrained(
            self.tmpdirname,
            bpe_vocab_file=self.bpe_vocab_file,
            bpe_json_file=self.bpe_json_file,
            use_bpe_encoder=self.use_bpe_encoder,
            unk_token="<unk>",
            **kwargs,
        )
        return tokenizer

    def get_input_output_texts(self, tokenizer):
        input_text = " lower"
        output_text = "\u0120lower"
        return input_text, output_text

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5) -> Tuple[str, list]:
        toks = [
            (i, tokenizer.decode([i], clean_up_tokenization_spaces=False))
            for i in range(len(tokenizer.bpe_tokenizer.encoder))
        ]
        toks = list(
            filter(
                lambda t: [t[0]]
                == tokenizer.encode(t[1], return_token_type_ids=None, add_special_tokens=False)["input_ids"],
                toks,
            )
        )
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        if min_length is not None and len(toks) < min_length and len(toks) > 0:
            while len(toks) < min_length:
                toks = toks + toks
        # toks_str = [t[1] for t in toks]
        toks_ids = [t[0] for t in toks]

        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt, return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        return output_txt, output_ids

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        text = " lower newer"
        bpe_tokens = ["\u0120low", "er", "\u0120", "n", "e", "w", "er"]

        # test tokenize
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        # test encode
        token_ids = tokenizer.encode(text)["input_ids"]

        # test decode
        decode_text = tokenizer.decode(token_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
        self.assertEqual(text, decode_text)

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [14, 15, 10, 9, 3, 2, 15])

    def test_internal_consistency(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, output_text = self.get_input_output_texts(tokenizer)

                tokens = tokenizer.tokenize(input_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                tokens_2 = tokenizer.convert_ids_to_tokens(ids)

                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_2 = tokenizer.encode(input_text, return_token_type_ids=None, add_special_tokens=False)["input_ids"]
                self.assertListEqual(ids, ids_2)

                tokens_2 = tokenizer.convert_ids_to_tokens(ids)
                self.assertNotEqual(len(tokens_2), 0)
                text_2 = tokenizer.decode(ids)
                self.assertIsInstance(text_2, str)

    def test_chinese(self):
        tokenizer = BasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("ah\u535A\u63A8zz"), ["ah", "\u535A", "\u63A8", "zz"])

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual(
            [tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]],
            [["T", "e", "s", "t"], ["Â", "Ń"], ["t", "e", "s", "t"]],
        )

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("skep_ernie_1.0_large_ch")

        text = tokenizer.encode("sequence builders", return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build", return_token_type_ids=None, add_special_tokens=False)[
            "input_ids"
        ]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [
            tokenizer.sep_token_id
        ]

    def test_pretokenized_inputs(self):
        # Test when inputs are pretokenized

        tokenizers = self.get_tokenizers(do_lower_case=False)  # , add_prefix_space=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                if hasattr(tokenizer, "add_prefix_space") and not tokenizer.add_prefix_space:
                    continue

                # Prepare a sequence from our tokenizer vocabulary
                sequence, ids = self.get_clean_sequence(tokenizer, with_prefix_space=True, max_length=20)

                # Test encode for pretokenized inputs
                output_sequence = tokenizer.encode(sequence, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]
                self.assertEqual(ids, output_sequence)

    def test_conversion_reversible(self):
        self.skipTest("bpe vocab not supported cls_token, bos_token")

    def test_offsets_mapping(self):
        self.skipTest("using basic-tokenizer or word-piece tokenzier to do this test, so to skpt")

    def test_special_tokens_mask_input_pairs(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = " lower"
                sequence_1 = "newer"
                encoded_sequence = tokenizer.encode(sequence_0, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]
                encoded_sequence += tokenizer.encode(sequence_1, return_token_type_ids=None, add_special_tokens=False)[
                    "input_ids"
                ]
                encoded_sequence_dict = tokenizer.encode(
                    sequence_0,
                    sequence_1,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                    # add_prefix_space=False,
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [
                    (x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)
                ]
                filtered_sequence = [x for x in filtered_sequence if x is not None]
                self.assertEqual(encoded_sequence, filtered_sequence)


class SkepWordPieceTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = SkepTokenizer
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english
    test_seq2seq = True
    use_bpe_encoder = False
    from_pretrained_kwargs = {"do_lower_case": False}

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
        ]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("\n".join(vocab_tokens))

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


class SkepChineseTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = SkepTokenizer
    space_between_special_tokens = False
    from_pretrained_filter = filter_non_english
    test_seq2seq = True
    use_bpe_encoder = False

    only_english_character = False

    def setUp(self):
        super().setUp()
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")

        shutil.copyfile(get_tests_dir("fixtures/vocab.zh.txt"), self.vocab_file)

        self.bpe_vocab_file = None
        self.bpe_json_file = None

    def get_tokenizer(self, **kwargs):
        return self.tokenizer_class.from_pretrained(
            self.tmpdirname,
            vocab_file=self.vocab_file,
            bpe_vocab_file=self.bpe_vocab_file,
            bpe_json_file=self.bpe_json_file,
            use_bpe_encoder=self.use_bpe_encoder,
            **kwargs,
        )

    def get_input_output_texts(self, tokenizer):
        input_text = "飞\u6868深度学习框架"
        output_text = "飞 桨 深 度 学 习 框 架"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize("飞\u6868深度学习框架")
        self.assertListEqual(tokens, list("飞桨深度学习框架"))

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [11, 12, 13, 10, 14, 15, 16, 17])

    def test_chinese(self):
        tokenizer = BasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("飞\u535A\u63A8桨"), ["飞", "\u535A", "\u63A8", "桨"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=False)
        tokens = tokenizer.tokenize(" \t飞!桨  \n 深度学 习  ")
        self.assertListEqual(tokens, ["飞", "!", "桨", "深", "度", "学", "习"])

    def test_basic_tokenizer_respects_never_split_tokens(self):
        tokenizer = BasicTokenizer(do_lower_case=False, never_split=["[UNK]"])

        tokens = tokenizer.tokenize(" \t飞!桨  \n 深度学 习  [UNK]")
        self.assertListEqual(tokens, ["飞", "!", "桨", "深", "度", "学", "习", "[UNK]"])

    def test_offsets_mapping(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                text = "这世界很美"
                pair = "我们需要共同守护"

                # No pair
                tokens_with_offsets = tokenizer.encode(
                    text, return_special_tokens_mask=True, return_offsets_mapping=True, add_special_tokens=True
                )
                added_tokens = tokenizer.num_special_tokens_to_add(False)
                offsets = tokens_with_offsets["offset_mapping"]

                # Assert there is the same number of tokens and offsets
                self.assertEqual(len(offsets), len(tokens_with_offsets["input_ids"]))

                # Assert there is online added_tokens special_tokens
                self.assertEqual(sum(tokens_with_offsets["special_tokens_mask"]), added_tokens)

                # Pairs
                tokens_with_offsets = tokenizer.encode(
                    text, pair, return_special_tokens_mask=True, return_offsets_mapping=True, add_special_tokens=True
                )
                added_tokens = tokenizer.num_special_tokens_to_add(True)
                offsets = tokens_with_offsets["offset_mapping"]

                # Assert there is the same number of tokens and offsets
                self.assertEqual(len(offsets), len(tokens_with_offsets["input_ids"]))

                # Assert there is online added_tokens special_tokens
                self.assertEqual(sum(tokens_with_offsets["special_tokens_mask"]), added_tokens)

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual([tokenizer.tokenize(t) for t in ["鲲", "\xad", "鹏"]], [["[UNK]"], [], ["[UNK]"]])

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("skep_ernie_1.0_large_ch")

        text = tokenizer.encode("sequence builders", return_token_type_ids=None, add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build", return_token_type_ids=None, add_special_tokens=False)[
            "input_ids"
        ]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [
            tokenizer.sep_token_id
        ]

    @slow
    def test_offsets_with_special_characters(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                sentence = f"北京的首都 {tokenizer.mask_token} 是北京"
                tokens = tokenizer.encode(
                    sentence,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                    spaces_between_special_tokens=self.space_between_special_tokens,
                )

                expected_results = [
                    ((0, 0), tokenizer.cls_token),
                    ((0, 1), "北"),
                    ((1, 2), "京"),
                    ((2, 3), "的"),
                    ((3, 4), "首"),
                    ((4, 5), "都"),
                    ((6, 12), "[MASK]"),
                    ((13, 14), "是"),
                    ((14, 15), "北"),
                    ((15, 16), "京"),
                    ((0, 0), tokenizer.sep_token),
                ]
                self.assertEqual(
                    [e[1] for e in expected_results], tokenizer.convert_ids_to_tokens(tokens["input_ids"])
                )

                self.assertEqual([e[0] for e in expected_results], tokens["offset_mapping"])

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

                # not yet supported in bert tokenizer
                """
                kwargs["tokenize_chinese_chars"] = False
                tokenizer = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer.encode(text_with_chinese_char, return_token_type_ids=None,add_special_tokens=False)["input_ids"]

                tokens_without_spe_char_p = tokenizer.convert_ids_to_tokens(ids_without_spe_char_p)

                # it is expected that only the first Chinese character is not preceded by "##".
                expected_tokens = [
                    f"##{token}" if idx != 0 else token for idx, token in enumerate(list_of_commun_chinese_char)
                ]
                self.assertListEqual(tokens_without_spe_char_p, expected_tokens)
                """

    def test_pretrained_model_lists(self):
        self.skipTest("`max_model_input_sizes` not found, so skip this test")
