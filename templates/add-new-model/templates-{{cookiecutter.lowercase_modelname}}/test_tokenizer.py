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

import os
import unittest

from paddlenlp.transformers.{{cookiecutter.lowercase_modelname}}.tokenizer import {{cookiecutter.camelcase_modelname}}Tokenizer

from ...testing_utils import slow
from ...transformers.test_tokenizer_common import TokenizerTesterMixin, filter_non_english

{%- if cookiecutter.tokenizer_type == "Based on Bert" %}
class {{cookiecutter.camelcase_modelname}}TokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = {{cookiecutter.camelcase_modelname}}Tokenizer
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english
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
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]

        self.vocab_file = os.path.join(
            self.tmpdirname, {{cookiecutter.camelcase_modelname}}Tokenizer.resource_files_names["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00E9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file)

        tokens = tokenizer.tokenize("UNwant\u00E9d,running")
        self.assertListEqual(tokens,
                             ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens),
                             [9, 6, 7, 12, 10, 11])

    def test_chinese(self):
        tokenizer = BasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("ah\u535A\u63A8zz"),
                             ["ah", "\u535A", "\u63A8", "zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "),
                             ["hello", "!", "how", "are", "you", "?"])
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=False)

        self.assertListEqual(tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "),
                             ["hällo", "!", "how", "are", "you", "?"])
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["h\u00E9llo"])

    def test_basic_tokenizer_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "),
                             ["hallo", "!", "how", "are", "you", "?"])
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_lower_strip_accents_default(self):
        tokenizer = BasicTokenizer(do_lower_case=True)

        self.assertListEqual(tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "),
                             ["hallo", "!", "how", "are", "you", "?"])
        self.assertListEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = BasicTokenizer(do_lower_case=False)

        self.assertListEqual(tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "),
                             ["HeLLo", "!", "how", "Are", "yoU", "?"])

    def test_basic_tokenizer_no_lower_strip_accents_false(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=False)

        self.assertListEqual(tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "),
                             ["HäLLo", "!", "how", "Are", "yoU", "?"])

    def test_basic_tokenizer_no_lower_strip_accents_true(self):
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=True)

        self.assertListEqual(tokenizer.tokenize(" \tHäLLo!how  \n Are yoU?  "),
                             ["HaLLo", "!", "how", "Are", "yoU", "?"])

    def test_basic_tokenizer_respects_never_split_tokens(self):
        tokenizer = BasicTokenizer(do_lower_case=False, never_split=["[UNK]"])

        self.assertListEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU? [UNK]"),
            ["HeLLo", "!", "how", "Are", "yoU", "?", "[UNK]"])

    def test_wordpiece_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un",
            "runn", "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(tokenizer.tokenize("unwanted running"),
                             ["un", "##want", "##ed", "runn", "##ing"])

        self.assertListEqual(tokenizer.tokenize("unwantedX running"),
                             ["[UNK]", "runn", "##ing"])

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual(
            [tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]],
            [["[UNK]"], [], ["[UNK]"]])

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("bert-base-uncased")

        text = tokenizer.encode("sequence builders",
                                return_token_type_ids=None,
                                add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build",
                                  return_token_type_ids=None,
                                  add_special_tokens=False)["input_ids"]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [101] + text + [102]
        assert encoded_pair == [101] + text + [102] + text_2 + [102]

    def test_offsets_with_special_characters(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(
                    f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, **kwargs)

                sentence = f"A, naïve {tokenizer.mask_token} AllenNLP sentence."
                tokens = tokenizer.encode(
                    sentence,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

                do_lower_case = tokenizer.do_lower_case if hasattr(
                    tokenizer, "do_lower_case") else False
                expected_results = ([
                    ((0, 0), tokenizer.cls_token),
                    ((0, 1), "A"),
                    ((1, 2), ","),
                    ((3, 5), "na"),
                    ((5, 6), "##ï"),
                    ((6, 8), "##ve"),
                    ((9, 15), tokenizer.mask_token),
                    ((16, 21), "Allen"),
                    ((21, 23), "##NL"),
                    ((23, 24), "##P"),
                    ((25, 33), "sentence"),
                    ((33, 34), "."),
                    ((0, 0), tokenizer.sep_token),
                ] if not do_lower_case else [
                    ((0, 0), tokenizer.cls_token),
                    ((0, 1), "a"),
                    ((1, 2), ","),
                    ((3, 8), "naive"),
                    ((9, 15), tokenizer.mask_token),
                    ((16, 21), "allen"),
                    ((21, 23), "##nl"),
                    ((23, 24), "##p"),
                    ((25, 33), "sentence"),
                    ((33, 34), "."),
                    ((0, 0), tokenizer.sep_token),
                ])

                self.assertEqual([e[1] for e in expected_results],
                                 tokenizer.convert_ids_to_tokens(
                                     tokens["input_ids"]))
                self.assertEqual([e[0] for e in expected_results],
                                 tokens["offset_mapping"])

    def test_change_tokenize_chinese_chars(self):
        list_of_commun_chinese_char = ["的", "人", "有"]
        text_with_chinese_char = "".join(list_of_commun_chinese_char)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(
                    f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                kwargs["tokenize_chinese_chars"] = True
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer.encode(
                    text_with_chinese_char,
                    return_token_type_ids=None,
                    add_special_tokens=False)["input_ids"]

                tokens_without_spe_char_p = tokenizer.convert_ids_to_tokens(
                    ids_without_spe_char_p)

                # it is expected that each Chinese character is not preceded by "##"
                self.assertListEqual(tokens_without_spe_char_p,
                                     list_of_commun_chinese_char)
{%- elif cookiecutter.tokenizer_type == "Based on BPETokenizer" %}

class {{cookiecutter.camelcase_modelname}}TokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = {{cookiecutter.camelcase_modelname}}Tokenizer
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
        self.vocab_file = os.path.join(self.tmpdirname, 'vocab.txt')
        with open(self.vocab_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab + ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]))

        # save bpe related files
        self.bpe_json_file = os.path.join(self.tmpdirname, 'encoder.json')
        self.bpe_vocab_file = os.path.join(self.tmpdirname, 'merges.txt')
        shutil.copyfile(get_tests_dir("fixtures/bpe.en/vocab.json"),
                        self.bpe_json_file)

        shutil.copyfile(get_tests_dir("fixtures/bpe.en/merges.txt"),
                        self.bpe_vocab_file)

    def get_tokenizer(self, **kwargs):
        tokenizer = self.tokenizer_class.from_pretrained(
            self.tmpdirname,
            bpe_vocab_file=self.bpe_vocab_file,
            bpe_json_file=self.bpe_json_file,
            use_bpe_encoder=self.use_bpe_encoder,
            unk_token='<unk>',
            **kwargs)
        return tokenizer

    def get_input_output_texts(self, tokenizer):
        input_text = " lower"
        output_text = "\u0120lower"
        return input_text, output_text

    def get_clean_sequence(self,
                           tokenizer,
                           with_prefix_space=False,
                           max_length=20,
                           min_length=5) -> Tuple[str, list]:
        toks = [(i, tokenizer.decode([i], clean_up_tokenization_spaces=False))
                for i in range(len(tokenizer.bpe_tokenizer.encoder))]
        toks = list(
            filter(
                lambda t: [t[0]] == tokenizer.encode(
                    t[1], return_token_type_ids=None, add_special_tokens=False)[
                        'input_ids'], toks))
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        if min_length is not None and len(toks) < min_length and len(toks) > 0:
            while len(toks) < min_length:
                toks = toks + toks
        # toks_str = [t[1] for t in toks]
        toks_ids = [t[0] for t in toks]

        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids,
                                      clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (tokenizer.decode(
                [toks_ids[0]], clean_up_tokenization_spaces=False) + " " +
                          tokenizer.decode(toks_ids[1:],
                                           clean_up_tokenization_spaces=False))
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt,
                                      return_token_type_ids=None,
                                      add_special_tokens=False)['input_ids']
        return output_txt, output_ids

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        text = " lower newer"
        bpe_tokens = ["\u0120low", "er", "\u0120", "n", "e", "w", "er"]

        # test tokenize
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        # test encode
        token_ids = tokenizer.encode(text)['input_ids']

        # test decode
        decode_text = tokenizer.decode(token_ids,
                                       skip_special_tokens=True,
                                       spaces_between_special_tokens=False)
        self.assertEqual(text, decode_text)

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens),
                             [14, 15, 10, 9, 3, 2, 15])

    def test_internal_consistency(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, output_text = self.get_input_output_texts(tokenizer)

                tokens = tokenizer.tokenize(input_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                tokens_2 = tokenizer.convert_ids_to_tokens(ids)

                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_2 = tokenizer.encode(input_text,
                                         return_token_type_ids=None,
                                         add_special_tokens=False)['input_ids']
                self.assertListEqual(ids, ids_2)

                tokens_2 = tokenizer.convert_ids_to_tokens(ids)
                self.assertNotEqual(len(tokens_2), 0)
                text_2 = tokenizer.decode(ids)
                self.assertIsInstance(text_2, str)

    def test_chinese(self):
        tokenizer = BasicTokenizer()

        self.assertListEqual(tokenizer.tokenize("ah\u535A\u63A8zz"),
                             ["ah", "\u535A", "\u63A8", "zz"])

    def test_clean_text(self):
        tokenizer = self.get_tokenizer()

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual(
            [tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]],
            [['T', 'e', 's', 't'], ['Â', 'Ń'], ['t', 'e', 's', 't']])

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained(
            "skep_ernie_1.0_large_ch")

        text = tokenizer.encode("sequence builders",
                                return_token_type_ids=None,
                                add_special_tokens=False)["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build",
                                  return_token_type_ids=None,
                                  add_special_tokens=False)["input_ids"]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id
                                    ] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [
            tokenizer.sep_token_id
        ] + text_2 + [tokenizer.sep_token_id]

    def test_pretokenized_inputs(self):
        # Test when inputs are pretokenized

        tokenizers = self.get_tokenizers(
            do_lower_case=False)  # , add_prefix_space=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                if hasattr(
                        tokenizer,
                        "add_prefix_space") and not tokenizer.add_prefix_space:
                    continue

                # Prepare a sequence from our tokenizer vocabulary
                sequence, ids = self.get_clean_sequence(tokenizer,
                                                        with_prefix_space=True,
                                                        max_length=20)
                # sequence_no_prefix_space = sequence.strip()
                token_sequence = sequence.split()
                # Test encode for pretokenized inputs
                output_sequence = tokenizer.encode(
                    sequence,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                self.assertEqual(ids, output_sequence)

    def test_conversion_reversible(self):
        self.skipTest("bpe vocab not supported cls_token, bos_token")

    def test_offsets_mapping(self):
        self.skipTest(
            "using basic-tokenizer or word-piece tokenzier to do this test, so to skip this testcase"
        )

    def test_special_tokens_mask_input_pairs(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = " lower"
                sequence_1 = "newer"
                encoded_sequence = tokenizer.encode(
                    sequence_0,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                encoded_sequence += tokenizer.encode(
                    sequence_1,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                encoded_sequence_dict = tokenizer.encode(
                    sequence_0,
                    sequence_1,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                    # add_prefix_space=False,
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict[
                    "special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask),
                                 len(encoded_sequence_w_special))

                filtered_sequence = [
                    (x if not special_tokens_mask[i] else None)
                    for i, x in enumerate(encoded_sequence_w_special)
                ]
                filtered_sequence = [
                    x for x in filtered_sequence if x is not None
                ]
                self.assertEqual(encoded_sequence, filtered_sequence)

{%- elif cookiecutter.tokenizer_type == "Based on SentencePiece" %}

# TODO(cookiecutter): to be tested under the actual case
class {{cookiecutter.camelcase_modelname}}TokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = {{cookiecutter.camelcase_modelname}}Tokenizer
    from_pretrained_vocab_key = "sentencepiece_model_file"
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True
    test_offsets = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = AlbertEnglishTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def test_convert_token_and_id(self):
        """Test ``_convert_token_to_id`` and ``_convert_id_to_token``."""
        token = "<pad>"
        token_id = 0

        self.assertEqual(self.get_tokenizer()._convert_token_to_id(token),
                         token_id)
        self.assertEqual(self.get_tokenizer()._convert_id_to_token(token_id),
                         token)

    def test_get_vocab(self):
        vocab_keys = list(self.get_tokenizer().get_vocab().keys())

        self.assertEqual(vocab_keys[0], "<pad>")
        self.assertEqual(vocab_keys[1], "<unk>")
        self.assertEqual(vocab_keys[-1], "▁eloquent")
        self.assertEqual(len(vocab_keys), 30_000)

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 30_000)

    def test_full_tokenizer(self):
        tokenizer = AlbertEnglishTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁this", "▁is", "▁a", "▁test"])

        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens),
                             [48, 25, 21, 1289])

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(tokens, [
            "▁i", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this",
            "▁is", "▁fal", "s", "é", "."
        ])
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids, [31, 23, 386, 19, 561, 3050, 15, 17, 48, 25, 8256, 18, 1, 9])

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                "▁i", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and",
                "▁this", "▁is", "▁fal", "s", "<unk>", "."
            ],
        )

    def test_sequence_builders(self):
        tokenizer = AlbertEnglishTokenizer(SAMPLE_VOCAB)

        text = tokenizer.encode("sequence builders")["input_ids"]
        text_2 = tokenizer.encode("multi-sequence build")["input_ids"]

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id
                                    ] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [
            tokenizer.sep_token_id
        ] + text_2 + [tokenizer.sep_token_id]

    @slow
    def test_tokenizer_integration(self):
        # fmt: off
        expected_encoding = {
            'attention_mask': [[
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ],
                               [
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                               ],
                               [
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                               ]],
            'input_ids': [[
                2, 21970, 13, 5, 6092, 167, 28, 7103, 2153, 673, 8, 7028, 12051,
                18, 17, 7103, 2153, 673, 8, 3515, 18684, 8, 4461, 6, 1927, 297,
                8, 12060, 2607, 18, 13, 5, 4461, 15, 10538, 38, 8, 135, 15, 822,
                58, 15, 993, 10363, 15, 1460, 8005, 4461, 15, 993, 255, 2328, 9,
                9, 9, 6, 26, 1112, 816, 3260, 13, 5, 103, 2377, 6, 17, 1112,
                816, 2782, 13, 5, 103, 10641, 6, 29, 84, 2512, 2430, 782, 18684,
                2761, 19, 808, 2430, 2556, 17, 855, 1480, 9477, 4091, 128,
                11712, 15, 7103, 2153, 673, 17, 24883, 9990, 9, 3
            ],
                          [
                              2, 11502, 25, 1006, 20, 782, 8, 11809, 855, 1732,
                              19393, 18667, 37, 367, 21018, 69, 1854, 34, 11860,
                              19124, 27, 156, 225, 17, 193, 4141, 19, 65, 9124,
                              9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0
                          ],
                          [
                              2, 14, 2231, 886, 2385, 17659, 84, 14, 16792,
                              1952, 9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0
                          ]],
            'token_type_ids': [[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
                               [
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                               ],
                               [
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                               ]]
        }

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="albert-base-v2",
        )

{% endif %}
