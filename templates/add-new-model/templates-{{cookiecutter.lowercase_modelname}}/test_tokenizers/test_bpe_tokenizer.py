# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
