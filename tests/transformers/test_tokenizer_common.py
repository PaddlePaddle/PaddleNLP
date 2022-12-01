# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 HuggingFace Inc.
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

import inspect
import itertools
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
import unittest
from collections import OrderedDict
from itertools import takewhile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from paddlenlp.transformers import (AlbertTokenizer, AutoTokenizer,
                                    BertTokenizer, PretrainedTokenizer)
from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase
from paddlenlp.transformers.tokenizer_utils import AddedToken, Trie
from ..testing_utils import get_tests_dir, slow

sys.path.append(str(Path(__file__).parent.parent / "utils"))

NON_ENGLISH_TAGS = [
    "chinese", "dutch", "french", "finnish", "german", "multilingual"
]

SMALL_TRAINING_CORPUS = [
    ["This is the first sentence.", "This is the second one."],
    [
        "This sentence (contains #) over symbols and numbers 12 3.",
        "But not this one."
    ],
]


def filter_non_english(_, pretrained_name: str):
    """Filter all the model for non-english language"""
    return not any([lang in pretrained_name for lang in NON_ENGLISH_TAGS])


def filter_roberta_detectors(_, pretrained_name: str):
    return "detector" not in pretrained_name


class TokenizerTesterMixin:

    tokenizer_class = None
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    from_pretrained_filter = None
    from_pretrained_vocab_key = "vocab_file"

    # set to True to test a sentencepiece tokenizer
    test_sentencepiece = False
    # set to True to ignore casing when testing a sentencepiece tokenizer
    # test_sentencepiece must also be set to True
    test_sentencepiece_ignore_case = False
    test_offsets = True

    only_english_character: bool = True

    def setUp(self) -> None:

        tokenizers_list = [(
            self.tokenizer_class,
            pretrained_name,
            self.from_pretrained_kwargs
            if self.from_pretrained_kwargs is not None else {},
        ) for pretrained_name in
                           self.tokenizer_class.pretrained_resource_files_map[
                               self.from_pretrained_vocab_key].keys()
                           if self.from_pretrained_filter is None or (
                               self.from_pretrained_filter is not None
                               and self.from_pretrained_filter(pretrained_name))
                           ]
        self.tokenizers_list = tokenizers_list[:1]

        with open(f"{get_tests_dir()}/sample_text.txt",
                  encoding="utf-8") as f_data:
            self._data = f_data.read().replace("\n\n", "\n").strip()

        self.tmpdirname = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_txt = self.get_clean_sequence(tokenizer)[0]
        return input_txt, input_txt

    def get_clean_sequence(self,
                           tokenizer,
                           with_prefix_space=False,
                           max_length=20,
                           min_length=5) -> Tuple[str, list]:
        toks = [(i, tokenizer.decode([i], clean_up_tokenization_spaces=False))
                for i in range(len(tokenizer))]

        # filter the english only character
        if self.only_english_character:
            toks = list(filter(lambda t: re.match(r"^[ a-zA-Z]+$", t[1]), toks))

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

    def get_tokenizers(self, **kwargs) -> List[PretrainedTokenizerBase]:
        return [self.get_tokenizer(**kwargs)]

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def tokenizer_integration_test_util(
        self,
        expected_encoding: Dict,
        model_name: str,
        sequences: List[str] = None,
        decode_kwargs: Dict[str, Any] = None,
        padding: bool = True,
    ):
        """
        Util for integration test.

        Text is tokenized and then reverted back to text. Both results are then checked.

        Args:
            expected_encoding:
                The expected result of the tokenizer output.
            model_name:
                The model name of the tokenizer to load and use.
            sequences:
                Can overwrite the texts that are used to check the tokenizer.
                This is useful if the tokenizer supports non english languages
                like france.
            decode_kwargs:
                Additional args for the ``decode`` function which reverts the
                tokenized text back to a string.
            padding:
                Activates and controls padding of the tokenizer.
        """
        decode_kwargs = {} if decode_kwargs is None else decode_kwargs

        if sequences is None:
            sequences = [
                "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides "
                "general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural "
                "Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained "
                "models in 100+ languages and deep interoperability between Jax, PyTorch and TensorFlow.",
                "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
                "conditioning on both left and right context in all layers.",
                "The quick brown fox jumps over the lazy dog.",
            ]

        if self.test_sentencepiece_ignore_case:
            sequences = [sequence.lower() for sequence in sequences]

        tokenizer_classes = [self.tokenizer_class]

        for tokenizer_class in tokenizer_classes:
            tokenizer = tokenizer_class.from_pretrained(model_name)

            encoding = tokenizer(sequences,
                                 padding=padding,
                                 return_attention_mask=True)
            decoded_sequences = [
                tokenizer.decode(seq, skip_special_tokens=True, **decode_kwargs)
                for seq in encoding["input_ids"]
            ]

            encoding_data = encoding.data
            self.assertDictEqual(encoding_data, expected_encoding)

            for expected, decoded in zip(sequences, decoded_sequences):
                if self.test_sentencepiece_ignore_case:
                    expected = expected.lower()
                self.assertEqual(expected, decoded)

    def assert_padded_input_match(self, input_r: list, input_p: list,
                                  max_length: int, pad_token_id: int):
        # Ensure we match max_length
        self.assertEqual(len(input_r), max_length)
        self.assertEqual(len(input_p), max_length)

        # Ensure the number of padded tokens is the same
        padded_tokens_r = list(
            takewhile(lambda i: i == pad_token_id, reversed(input_r)))
        padded_tokens_p = list(
            takewhile(lambda i: i == pad_token_id, reversed(input_p)))
        self.assertSequenceEqual(padded_tokens_r, padded_tokens_p)

    def assert_batch_padded_input_match(
        self,
        input_r: dict,
        input_p: dict,
        max_length: int,
        pad_token_id: int,
        model_main_input_name: str = "input_ids",
    ):
        for i_r in input_r.values():
            self.assertEqual(len(i_r), 2), self.assertEqual(
                len(i_r[0]),
                max_length), self.assertEqual(len(i_r[1]), max_length)
            self.assertEqual(len(i_r), 2), self.assertEqual(
                len(i_r[0]),
                max_length), self.assertEqual(len(i_r[1]), max_length)

        for i_r, i_p in zip(input_r[model_main_input_name],
                            input_p[model_main_input_name]):
            self.assert_padded_input_match(i_r, i_p, max_length, pad_token_id)

        for i_r, i_p in zip(input_r["attention_mask"],
                            input_p["attention_mask"]):
            self.assertSequenceEqual(i_r, i_p)

    @staticmethod
    def convert_batch_encode_plus_format_to_encode_plus(
            batch_encode_plus_sequences):
        # Switch from batch_encode_plus format:   {'input_ids': [[...], [...]], ...}
        # to the list of examples/ encode_plus format: [{'input_ids': [...], ...}, {'input_ids': [...], ...}]
        return [{
            value: batch_encode_plus_sequences[value][i]
            for value in batch_encode_plus_sequences.keys()
        } for i in range(len(batch_encode_plus_sequences["input_ids"]))]

    # TODO: this test can be combined with `test_sentencepiece_tokenize_and_convert_tokens_to_string` after the latter is extended to all tokenizers.
    def test_tokenize_special_tokens(self):
        """Test `tokenize` with special tokens."""
        tokenizers = self.get_tokenizers(do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                SPECIAL_TOKEN_1 = "[SPECIAL_TOKEN_1]"
                SPECIAL_TOKEN_2 = "[SPECIAL_TOKEN_2]"

                # TODO:
                # Can we combine `unique_no_split_tokens` and `all_special_tokens`(and properties related to it)
                # with one variable(property) for a better maintainability?

                # `add_tokens` method stores special tokens only in `tokenizer.unique_no_split_tokens`. (in tokenization_utils.py)
                tokenizer.add_tokens([SPECIAL_TOKEN_1], special_tokens=True)
                # `add_special_tokens` method stores special tokens in `tokenizer.additional_special_tokens`,
                # which also occur in `tokenizer.all_special_tokens`. (in tokenization_utils_base.py)
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": [SPECIAL_TOKEN_2]})

                token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
                token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)

                self.assertEqual(len(token_1), 1)
                self.assertEqual(len(token_2), 1)
                self.assertEqual(token_1[0], SPECIAL_TOKEN_1)
                self.assertEqual(token_2[0], SPECIAL_TOKEN_2)

    # TODO: this test could be extended to all tokenizers - not just the sentencepiece
    def test_sentencepiece_tokenize_and_convert_tokens_to_string(self):
        """Test ``_tokenize`` and ``convert_tokens_to_string``."""
        if not self.test_sentencepiece:
            return

        tokenizer = self.get_tokenizer()
        text = "This is text to test the tokenizer."

        if self.test_sentencepiece_ignore_case:
            text = text.lower()

        tokens = tokenizer.tokenize(text)

        self.assertTrue(len(tokens) > 0)

        # check if converting back to original text works
        reverse_text = tokenizer.convert_tokens_to_string(tokens)

        if self.test_sentencepiece_ignore_case:
            reverse_text = reverse_text.lower()

        self.assertEqual(reverse_text, text)

    def test_subword_regularization_tokenizer(self) -> None:
        if not self.test_sentencepiece:
            return

        # Subword regularization is only available for the slow tokenizer.
        sp_model_kwargs = {
            "enable_sampling": True,
            "alpha": 0.1,
            "nbest_size": -1
        }
        tokenizer = self.get_tokenizer(sp_model_kwargs=sp_model_kwargs)

        self.assertTrue(hasattr(tokenizer, "sp_model_kwargs"))
        self.assertIsNotNone(tokenizer.sp_model_kwargs)
        self.assertTrue(isinstance(tokenizer.sp_model_kwargs, dict))
        self.assertEqual(tokenizer.sp_model_kwargs, sp_model_kwargs)
        self.check_subword_sampling(tokenizer)

    def test_pickle_subword_regularization_tokenizer(self) -> None:
        if not self.test_sentencepiece:
            return
        """Google pickle __getstate__ __setstate__ if you are struggling with this."""
        # Subword regularization is only available for the slow tokenizer.
        sp_model_kwargs = {
            "enable_sampling": True,
            "alpha": 0.1,
            "nbest_size": -1
        }
        tokenizer = self.get_tokenizer(sp_model_kwargs=sp_model_kwargs)
        tokenizer_bin = pickle.dumps(tokenizer)
        del tokenizer
        tokenizer_new = pickle.loads(tokenizer_bin)

        self.assertTrue(hasattr(tokenizer_new, "sp_model_kwargs"))
        self.assertIsNotNone(tokenizer_new.sp_model_kwargs)
        self.assertTrue(isinstance(tokenizer_new.sp_model_kwargs, dict))
        self.assertEqual(tokenizer_new.sp_model_kwargs, sp_model_kwargs)
        self.check_subword_sampling(tokenizer_new)

    def test_save_sentencepiece_tokenizer(self) -> None:
        if not self.test_sentencepiece:
            return
        # We want to verify that we will be able to save the tokenizer even if the original files that were used to
        # build the tokenizer have been deleted in the meantime.
        text = "This is text to test the tokenizer."

        tokenizer_slow_1 = self.get_tokenizer()
        encoding_tokenizer_slow_1 = tokenizer_slow_1(text)

        tmpdirname_1 = tempfile.mkdtemp()
        tmpdirname_2 = tempfile.mkdtemp()

        tokenizer_slow_1.save_pretrained(tmpdirname_1)
        tokenizer_slow_2 = self.tokenizer_class.from_pretrained(tmpdirname_1)
        encoding_tokenizer_slow_2 = tokenizer_slow_2(text)

        shutil.rmtree(tmpdirname_1)
        tokenizer_slow_2.save_pretrained(tmpdirname_2)

        tokenizer_slow_3 = self.tokenizer_class.from_pretrained(tmpdirname_2)
        encoding_tokenizer_slow_3 = tokenizer_slow_3(text)
        shutil.rmtree(tmpdirname_2)

        self.assertEqual(encoding_tokenizer_slow_1, encoding_tokenizer_slow_2)
        self.assertEqual(encoding_tokenizer_slow_1, encoding_tokenizer_slow_3)

    def test_model_input_names_signature(self):
        accepted_model_main_input_names = [
            "input_ids",  # nlp models
            "input_values",  # speech models
        ]

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            # first name of model_input_names has to correspond to main model input name
            # to make sure `tokenizer.pad(...)` works correctly
            self.assertTrue(tokenizer.model_input_names[0] in
                            accepted_model_main_input_names)

    def test_tokenizer_slow_store_full_signature(self):
        signature = inspect.signature(self.tokenizer_class.__init__)
        tokenizer = self.get_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty:
                self.assertIn(parameter_name, tokenizer.init_kwargs)

    def test_tokenizers_common_properties(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                attributes_list = [
                    "bos_token",
                    "eos_token",
                    "unk_token",
                    "sep_token",
                    "pad_token",
                    "cls_token",
                    "mask_token",
                ]
                for attr in attributes_list:
                    self.assertTrue(hasattr(tokenizer, attr))
                    self.assertTrue(hasattr(tokenizer, attr + "_id"))

                self.assertTrue(hasattr(tokenizer, "additional_special_tokens"))
                self.assertTrue(
                    hasattr(tokenizer, "additional_special_tokens_ids"))

                attributes_list = [
                    "model_max_length",
                    "init_inputs",
                    "init_kwargs",
                ]

                attributes_list += [
                    "added_tokens_encoder",
                    "added_tokens_decoder",
                ]
                for attr in attributes_list:
                    self.assertTrue(hasattr(tokenizer, attr))

    def test_tokenizers_common_ids_setters(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                attributes_list = [
                    "bos_token",
                    "eos_token",
                    "unk_token",
                    "sep_token",
                    "pad_token",
                    "cls_token",
                    "mask_token",
                ]

                vocab = tokenizer.get_vocab()
                token_id_to_test_setters = next(iter(vocab.values()))
                token_to_test_setters = tokenizer.convert_ids_to_tokens(
                    token_id_to_test_setters, skip_special_tokens=False)

                for attr in attributes_list:
                    setattr(tokenizer, attr + "_id", None)
                    self.assertEqual(getattr(tokenizer, attr), None)
                    self.assertEqual(getattr(tokenizer, attr + "_id"), None)

                    setattr(tokenizer, attr + "_id", token_id_to_test_setters)
                    self.assertEqual(getattr(tokenizer, attr),
                                     token_to_test_setters)
                    self.assertEqual(getattr(tokenizer, attr + "_id"),
                                     token_id_to_test_setters)

                setattr(tokenizer, "additional_special_tokens_ids", [])
                self.assertListEqual(
                    getattr(tokenizer, "additional_special_tokens"), [])
                self.assertListEqual(
                    getattr(tokenizer, "additional_special_tokens_ids"), [])

                setattr(tokenizer, "additional_special_tokens_ids",
                        [token_id_to_test_setters])
                self.assertListEqual(
                    getattr(tokenizer, "additional_special_tokens"),
                    [token_to_test_setters])
                self.assertListEqual(
                    getattr(tokenizer, "additional_special_tokens_ids"),
                    [token_id_to_test_setters])

    def test_save_and_load_tokenizer(self):
        # safety check on max_len default value so we are sure the test works
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertNotEqual(tokenizer.model_max_length, 42)

        # Now let's start the test
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                before_tokens = tokenizer.encode(sample_text,
                                                 add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(
                    tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text,
                                                      add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens["input_ids"],
                                     after_tokens["input_ids"])
                self.assertDictEqual(before_vocab, after_vocab)

                shutil.rmtree(tmpdirname)

        tokenizers = self.get_tokenizers(model_max_length=42)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                tokenizer.add_tokens(["bim", "bambam"])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append("new_additional_special_token")
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": additional_special_tokens})
                before_tokens = tokenizer.encode(sample_text,
                                                 add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(
                    tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text,
                                                      add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens["input_ids"],
                                     after_tokens["input_ids"])
                self.assertDictEqual(before_vocab, after_vocab)
                self.assertIn("bim", after_vocab)
                self.assertIn("bambam", after_vocab)
                self.assertIn("new_additional_special_token",
                              after_tokenizer.additional_special_tokens)
                self.assertEqual(after_tokenizer.model_max_length, 42)

                tokenizer = tokenizer.__class__.from_pretrained(
                    tmpdirname, model_max_length=43)
                self.assertEqual(tokenizer.model_max_length, 43)

                shutil.rmtree(tmpdirname)

        # Test that we can also use the non-legacy saving format for fast tokenizers
        tokenizers = self.get_tokenizers(model_max_length=42)
        for tokenizer in tokenizers:
            if not tokenizer.is_fast:
                continue
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                tokenizer.add_tokens(["bim", "bambam"])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append("new_additional_special_token")
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": additional_special_tokens})
                before_tokens = tokenizer.encode(sample_text,
                                                 add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(
                    tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text,
                                                      add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)
                self.assertIn("bim", after_vocab)
                self.assertIn("bambam", after_vocab)
                self.assertIn("new_additional_special_token",
                              after_tokenizer.additional_special_tokens)
                self.assertEqual(after_tokenizer.model_max_length, 42)

                tokenizer = tokenizer.__class__.from_pretrained(
                    tmpdirname, model_max_length=43)
                self.assertEqual(tokenizer.model_max_length, 43)

                shutil.rmtree(tmpdirname)

    # def test_pickle_tokenizer(self):
    #     """Google pickle __getstate__ __setstate__ if you are struggling with this."""
    #     tokenizers = self.get_tokenizers()
    #     for tokenizer in tokenizers:
    #         with self.subTest(f"{tokenizer.__class__.__name__}"):
    #             self.assertIsNotNone(tokenizer)
    #
    #             text = "Munich and Berlin are nice cities"
    #             subwords = tokenizer.tokenize(text)
    #
    #             filename = os.path.join(self.tmpdirname, "tokenizer.bin")
    #             with open(filename, "wb") as handle:
    #                 pickle.dump(tokenizer, handle)
    #
    #             with open(filename, "rb") as handle:
    #                 tokenizer_new = pickle.load(handle)
    #
    #             subwords_loaded = tokenizer_new.tokenize(text)
    #
    #             self.assertListEqual(subwords, subwords_loaded)

    def test_pickle_added_tokens(self):
        tok1 = AddedToken("<s>",
                          rstrip=True,
                          lstrip=True,
                          normalized=False,
                          single_word=True)
        tok2 = pickle.loads(pickle.dumps(tok1))

        self.assertEqual(tok1.__getstate__(), tok2.__getstate__())

    def test_added_tokens_do_lower_case(self):
        tokenizers = self.get_tokenizers(do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if not hasattr(tokenizer,
                               "do_lower_case") or not tokenizer.do_lower_case:
                    continue

                special_token = tokenizer.all_special_tokens[0]

                text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
                text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

                toks_before_adding = tokenizer.tokenize(
                    text)  # toks before adding new_toks

                new_toks = [
                    "aaaaa bbbbbb", "cccccccccdddddddd", "AAAAA BBBBBB",
                    "CCCCCCCCCDDDDDDDD"
                ]
                added = tokenizer.add_tokens([
                    AddedToken(tok, lstrip=True, rstrip=True)
                    for tok in new_toks
                ])

                toks_after_adding = tokenizer.tokenize(text)
                toks_after_adding2 = tokenizer.tokenize(text2)

                # Rust tokenizers dont't lowercase added tokens at the time calling `tokenizer.add_tokens`,
                # while python tokenizers do, so new_toks 0 and 2 would be treated as the same, so do new_toks 1 and 3.
                self.assertIn(added, [2, 4])

                self.assertListEqual(toks_after_adding, toks_after_adding2)
                self.assertTrue(
                    len(toks_before_adding) >
                    len(toks_after_adding
                        ),  # toks_before_adding should be longer
                )

                # Check that none of the special tokens are lowercased
                sequence_with_special_tokens = "A " + " yEs ".join(
                    tokenizer.all_special_tokens) + " B"
                # Convert the tokenized list to str as some special tokens are tokenized like normal tokens
                # which have a prefix spacee e.g. the mask token of Albert, and cannot match the original
                # special tokens exactly.
                tokenized_sequence = "".join(
                    tokenizer.tokenize(sequence_with_special_tokens))

                for special_token in tokenizer.all_special_tokens:
                    self.assertTrue(special_token in tokenized_sequence)

        tokenizers = self.get_tokenizers(do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if hasattr(tokenizer,
                           "do_lower_case") and tokenizer.do_lower_case:
                    continue

                special_token = tokenizer.all_special_tokens[0]

                text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
                text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

                toks_before_adding = tokenizer.tokenize(
                    text)  # toks before adding new_toks

                new_toks = [
                    "aaaaa bbbbbb", "cccccccccdddddddd", "AAAAA BBBBBB",
                    "CCCCCCCCCDDDDDDDD"
                ]
                added = tokenizer.add_tokens([
                    AddedToken(tok, lstrip=True, rstrip=True)
                    for tok in new_toks
                ])
                self.assertIn(added, [2, 4])

                toks_after_adding = tokenizer.tokenize(text)
                toks_after_adding2 = tokenizer.tokenize(text2)

                self.assertEqual(
                    len(toks_after_adding),
                    len(toks_after_adding2))  # Length should still be the same
                self.assertNotEqual(
                    toks_after_adding[1], toks_after_adding2[1]
                )  # But at least the first non-special tokens should differ
                self.assertTrue(
                    len(toks_before_adding) >
                    len(toks_after_adding
                        ),  # toks_before_adding should be longer
                )

    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode(
                    "aaaaa bbbbbb low cccccccccdddddddd l",
                    return_token_type_ids=None,
                    add_special_tokens=False)["input_ids"]
                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

                new_toks_2 = {
                    "eos_token": ">>>>|||<||<<|<<",
                    "pad_token": "<<<<<|||>|>>>>|>"
                }
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l",
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    def test_add_special_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, ids = self.get_clean_sequence(tokenizer)

                special_token = "[SPECIAL_TOKEN]"

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token = tokenizer.encode(
                    special_token,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                self.assertEqual(len(encoded_special_token), 1)

                text = tokenizer.decode(ids + encoded_special_token,
                                        clean_up_tokenization_spaces=False)
                encoded = tokenizer.encode(
                    text, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']

                input_encoded = tokenizer.encode(
                    input_text,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                special_token_id = tokenizer.encode(
                    special_token,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                self.assertEqual(encoded, input_encoded + special_token_id)

                decoded = tokenizer.decode(encoded, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_internal_consistency(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, output_text = self.get_input_output_texts(tokenizer)

                tokens = tokenizer.tokenize(input_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_2 = tokenizer.encode(input_text,
                                         return_token_type_ids=None,
                                         add_special_tokens=False)['input_ids']
                self.assertListEqual(ids, ids_2)

                tokens_2 = tokenizer.convert_ids_to_tokens(ids)
                self.assertNotEqual(len(tokens_2), 0)
                text_2 = tokenizer.decode(ids)
                self.assertIsInstance(text_2, str)

                self.assertEqual(text_2, output_text)

    def test_encode_decode_with_spaces(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                new_toks = [
                    AddedToken("[ABC]", normalized=False),
                    AddedToken("[DEF]", normalized=False),
                    AddedToken("GHI IHG", normalized=False),
                ]
                tokenizer.add_tokens(new_toks)
                input = "[ABC][DEF][ABC]GHI IHG[DEF]"
                if self.space_between_special_tokens:
                    output = "[ABC] [DEF] [ABC] GHI IHG [DEF]"
                else:
                    output = input
                encoded = tokenizer.encode(
                    input, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                decoded = tokenizer.decode(encoded,
                                           spaces_between_special_tokens=self.
                                           space_between_special_tokens)
                self.assertIn(decoded, [output, output.lower()])

    def test_pretrained_model_lists(self):
        # We should have at least one default checkpoint for each tokenizer
        # We should specify the max input length as well (used in some part to list the pretrained checkpoints)
        self.assertGreaterEqual(
            len(self.tokenizer_class.pretrained_resource_files_map), 1)
        self.assertGreaterEqual(
            len(
                list(
                    self.tokenizer_class.pretrained_resource_files_map.values())
                [0]), 1)
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

        weights_list.sort()
        for weights_list_2 in weights_lists_2:
            weights_list_2.sort()
            self.assertListEqual(weights_list, weights_list_2)

    def test_mask_output(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                if (tokenizer.build_inputs_with_special_tokens.__qualname__.
                        split(".")[0] != "PretrainedTokenizer"
                        and "token_type_ids" in tokenizer.model_input_names):
                    seq_0 = "Test this method."
                    seq_1 = "With these inputs."
                    information = tokenizer.encode(seq_0,
                                                   seq_1,
                                                   add_special_tokens=True)
                    sequences, mask = information["input_ids"], information[
                        "token_type_ids"]
                    self.assertEqual(len(sequences), len(mask))

    def test_token_type_ids(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0 = "Test this method."

                # We want to have sequence 0 and sequence 1 are tagged
                # respectively with 0 and 1 token_ids
                # (regardless of whether the model use token type ids)
                # We use this assumption in the QA pipeline among other place
                output = tokenizer(seq_0, return_token_type_ids=True)
                self.assertIn(0, output["token_type_ids"])

    def test_number_of_added_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                seq_0 = "Test this method."
                seq_1 = "With these inputs."

                sequences = tokenizer.encode(
                    seq_0,
                    seq_1,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                attached_sequences = tokenizer.encode(
                    seq_0, seq_1, add_special_tokens=True)['input_ids']

                # Method is implemented (e.g. not GPT-2)
                if len(attached_sequences) != 2:
                    self.assertEqual(
                        tokenizer.num_special_tokens_to_add(pair=True),
                        len(attached_sequences) - len(sequences))

    def test_maximum_encoding_length_single_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False,
                                         model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)

                sequence = tokenizer.encode(
                    seq_0, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                total_length = len(sequence)

                self.assertGreater(
                    total_length, 4,
                    "Issue with the testing sequence, please update it it's too short"
                )

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_1 = seq_0 * model_max_length

                sequence1 = tokenizer(seq_1,
                                      return_token_type_ids=None,
                                      add_special_tokens=False,
                                      truncation=False)
                total_length1 = len(sequence1["input_ids"])
                self.assertGreater(
                    total_length1, model_max_length,
                    "Issue with the testing sequence, please update it it's too short"
                )

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token
                    and tokenizer.pad_token_id >= 0 else [False])
                for padding_state in padding_strategies:
                    with self.subTest(f"Padding: {padding_state}"):
                        for truncation_state in [
                                True, "longest_first", "only_first"
                        ]:
                            with self.subTest(
                                    f"Truncation: {truncation_state}"):
                                output = tokenizer(seq_1,
                                                   padding=padding_state,
                                                   truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"]),
                                                 model_max_length)

                                output = tokenizer([seq_1],
                                                   padding=padding_state,
                                                   truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"][0]),
                                                 model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("PaddleNLP",
                                             level="WARNING") as cm:
                            output = tokenizer(seq_1,
                                               padding=padding_state,
                                               truncation=False)
                            self.assertNotEqual(len(output["input_ids"]),
                                                model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(cm.records[0].message.startswith(
                            "Token indices sequence length is longer than the specified maximum sequence length for this model"
                        ))

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("PaddleNLP",
                                             level="WARNING") as cm:
                            output = tokenizer([seq_1],
                                               padding=padding_state,
                                               truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]),
                                                model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(cm.records[0].message.startswith(
                            "Token indices sequence length is longer than the specified maximum sequence length for this model"
                        ))

                # Overflowing tokens
                stride = 2
                information = tokenizer(
                    seq_0,
                    max_length=total_length - 2,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="longest_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                truncated_sequence = information["input_ids"]
                overflowing_tokens = information["overflowing_tokens"]

                self.assertEqual(len(truncated_sequence), total_length - 2)
                self.assertEqual(truncated_sequence, sequence[:-2])

                self.assertEqual(len(overflowing_tokens), 2 + stride)
                self.assertEqual(overflowing_tokens, sequence[-(2 + stride):])

    def test_maximum_encoding_length_pair_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False,
                                         model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Build a sequence from our model's vocabulary
                stride = 2
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)
                if len(ids) <= 2 + stride:
                    seq_0 = (seq_0 + " ") * (2 + stride)
                    ids = None

                seq0_tokens = tokenizer.encode(
                    seq_0, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                self.assertGreater(len(seq0_tokens), 2 + stride)

                seq_1 = "This is another sentence to be encoded."
                seq1_tokens = tokenizer.encode(
                    seq_1, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                if abs(len(seq0_tokens) - len(seq1_tokens)) <= 2:
                    seq1_tokens = seq1_tokens + seq1_tokens
                    seq_1 = tokenizer.decode(seq1_tokens,
                                             clean_up_tokenization_spaces=False)
                seq1_tokens = tokenizer.encode(
                    seq_1, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']

                self.assertGreater(len(seq1_tokens), 2 + stride)

                smallest = seq1_tokens if len(seq0_tokens) > len(
                    seq1_tokens) else seq0_tokens

                # We are not using the special tokens - a bit too hard to test all the tokenizers with this
                # TODO try this again later
                sequence = tokenizer.encode(
                    seq_0,
                    seq_1,
                    return_token_type_ids=None,
                    add_special_tokens=False)[
                        'input_ids']  # , add_prefix_space=False)

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_2 = seq_0 * model_max_length
                self.assertGreater(len(seq_2), model_max_length)

                sequence1 = tokenizer(seq_1,
                                      return_token_type_ids=None,
                                      add_special_tokens=False,
                                      truncation=False)
                total_length1 = len(sequence1["input_ids"])
                sequence2 = tokenizer(seq_2,
                                      seq_1,
                                      return_token_type_ids=None,
                                      add_special_tokens=False,
                                      truncation=False)
                total_length2 = len(sequence2["input_ids"])
                self.assertLess(
                    total_length1, model_max_length - 10,
                    "Issue with the testing sequence, please update it.")
                self.assertGreater(
                    total_length2, model_max_length,
                    "Issue with the testing sequence, please update it.")

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token
                    and tokenizer.pad_token_id >= 0 else [False])
                for padding_state in padding_strategies:
                    with self.subTest(
                            f"{tokenizer.__class__.__name__} Padding: {padding_state}"
                    ):
                        for truncation_state in [
                                True, "longest_first", "only_first"
                        ]:
                            with self.subTest(
                                    f"{tokenizer.__class__.__name__} Truncation: {truncation_state}"
                            ):
                                output = tokenizer(seq_2,
                                                   seq_1,
                                                   padding=padding_state,
                                                   truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"]),
                                                 model_max_length)

                                output = tokenizer([seq_2], [seq_1],
                                                   padding=padding_state,
                                                   truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"][0]),
                                                 model_max_length)

                        # Simple
                        output = tokenizer(seq_1,
                                           seq_2,
                                           padding=padding_state,
                                           truncation="only_second")
                        self.assertEqual(len(output["input_ids"]),
                                         model_max_length)

                        output = tokenizer([seq_1], [seq_2],
                                           padding=padding_state,
                                           truncation="only_second")
                        self.assertEqual(len(output["input_ids"][0]),
                                         model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("PaddleNLP",
                                             level="WARNING") as cm:
                            output = tokenizer(seq_1,
                                               seq_2,
                                               padding=padding_state,
                                               truncation=False)
                            self.assertNotEqual(len(output["input_ids"]),
                                                model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(cm.records[0].message.startswith(
                            "Token indices sequence length is longer than the specified maximum sequence length for this model"
                        ))

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("PaddleNLP",
                                             level="WARNING") as cm:
                            output = tokenizer([seq_1], [seq_2],
                                               padding=padding_state,
                                               truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]),
                                                model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(cm.records[0].message.startswith(
                            "Token indices sequence length is longer than the specified maximum sequence length for this model"
                        ))

                truncated_first_sequence = tokenizer.encode(
                    seq_0, return_token_type_ids=None, add_special_tokens=False
                )['input_ids'][:-2] + tokenizer.encode(
                    seq_1, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                truncated_second_sequence = (tokenizer.encode(
                    seq_0, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids'] + tokenizer.encode(
                        seq_1,
                        return_token_type_ids=None,
                        add_special_tokens=False)['input_ids'][:-2])
                truncated_longest_sequence = (
                    truncated_first_sequence
                    if len(seq0_tokens) > len(seq1_tokens) else
                    truncated_second_sequence)

                overflow_first_sequence = tokenizer.encode(
                    seq_0, return_token_type_ids=None, add_special_tokens=False
                )['input_ids'][-(2 + stride):] + tokenizer.encode(
                    seq_1, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                overflow_second_sequence = (tokenizer.encode(
                    seq_0, return_token_type_ids=None,
                    add_special_tokens=False)['input_ids'] + tokenizer.encode(
                        seq_1,
                        return_token_type_ids=None,
                        add_special_tokens=False)['input_ids'][-(2 + stride):])
                overflow_longest_sequence = (overflow_first_sequence if
                                             len(seq0_tokens) > len(seq1_tokens)
                                             else overflow_second_sequence)

                with self.assertRaises(ValueError) as context:
                    information = tokenizer(
                        seq_0,
                        seq_1,
                        max_length=len(sequence) - 2,
                        return_token_type_ids=None,
                        add_special_tokens=False,
                        stride=stride,
                        truncation="longest_first",
                        return_overflowing_tokens=True,
                        # add_prefix_space=False,
                    )

                self.assertTrue(context.exception.args[0].startswith(
                    "Not possible to return overflowing tokens for pair of sequences with the "
                    "`longest_first`. Please select another truncation strategy than `longest_first`, "
                    "for instance `only_second` or `only_first`."))

                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                # No overflowing tokens when using 'longest' in python tokenizers
                with self.assertRaises(ValueError) as context:
                    information = tokenizer(
                        seq_0,
                        seq_1,
                        max_length=len(sequence) - 2,
                        return_token_type_ids=None,
                        add_special_tokens=False,
                        stride=stride,
                        truncation=True,
                        return_overflowing_tokens=True,
                        # add_prefix_space=False,
                    )

                self.assertTrue(context.exception.args[0].startswith(
                    "Not possible to return overflowing tokens for pair of sequences with the "
                    "`longest_first`. Please select another truncation strategy than `longest_first`, "
                    "for instance `only_second` or `only_first`."))

                information_first_truncated = tokenizer(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                truncated_sequence = information_first_truncated["input_ids"]
                overflowing_tokens = information_first_truncated[
                    "overflowing_tokens"]

                self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                self.assertEqual(truncated_sequence, truncated_first_sequence)

                self.assertEqual(len(overflowing_tokens), 2 + stride)
                self.assertEqual(overflowing_tokens,
                                 seq0_tokens[-(2 + stride):])

                information_second_truncated = tokenizer(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    return_token_type_ids=None,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_second",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers

                truncated_sequence = information_second_truncated["input_ids"]
                overflowing_tokens = information_second_truncated[
                    "overflowing_tokens"]

                self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                self.assertEqual(truncated_sequence, truncated_second_sequence)

                self.assertEqual(len(overflowing_tokens), 2 + stride)
                self.assertEqual(overflowing_tokens,
                                 seq1_tokens[-(2 + stride):])

    # def test_encode_input_type(self):
    #     tokenizers = self.get_tokenizers(do_lower_case=False)
    #     for tokenizer in tokenizers:
    #         with self.subTest(f"{tokenizer.__class__.__name__}"):
    #             sequence = "Let's encode this sequence"

    #             tokens = sequence.split()  # tokenizer.tokenize(sequence)
    #             # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    #             formatted_input = tokenizer.encode(sequence, add_special_tokens=True, add_prefix_space=False)

    #             self.assertEqual(
    #                 tokenizer.encode(tokens, is_split_into_words=True, add_special_tokens=True), formatted_input
    #             )
    #             # This is not supported with the Rust tokenizers
    #             # self.assertEqual(tokenizer.encode(input_ids, add_special_tokens=True), formatted_input)

    # def test_swap_special_token(self):
    #     tokenizers = self.get_tokenizers(do_lower_case=False)
    #     for tokenizer in tokenizers:
    #         with self.subTest(f"{tokenizer.__class__.__name__}"):
    #             # Our mask token
    #             mask = "<mask>"
    #             # We take a single word in the middle of the vocabulary
    #             all_tokens = sorted(tokenizer.get_vocab().keys())
    #             word = tokenizer.decode(tokenizer.encode(all_tokens[len(all_tokens)//2], add_special_tokens=False)[:1])

    #             sequence_0 = "Encode " + word + " sequence"
    #             sequence_masked_0 = "Encode " + mask + " sequence"

    #             sequence_1 = word + " this sequence"
    #             sequence_masked_1 = mask + " this sequence"

    #             # Add tokens so that masked token isn't split
    #             # tokens = [AddedToken(t, lstrip=True, normalized=False) for t in sequence.split()]
    #             # tokenizer.add_tokens(tokens)
    #             tokenizer.add_special_tokens(
    #                 {"mask_token": AddedToken(mask, normalized=False)}
    #             )  # Eat left space on Byte-level BPE tokenizers
    #             mask_ind = tokenizer.convert_tokens_to_ids(mask)

    #             # Test first masked sequence
    #             encoded_0 = tokenizer.encode(sequence_0, add_special_tokens=False)
    #             encoded_masked = tokenizer.encode(sequence_masked_0, add_special_tokens=False)
    #             self.assertEqual(len(encoded_masked), len(encoded_0))
    #             mask_loc = encoded_masked.index(mask_ind)
    #             encoded_masked[mask_loc] = encoded_0[mask_loc]

    #             self.assertEqual(encoded_masked, encoded_0)

    #             # Test second masked sequence
    #             encoded_1 = tokenizer.encode(sequence_1, add_special_tokens=False)
    #             encoded_masked = tokenizer.encode(sequence_masked_1, add_special_tokens=False)
    #             self.assertEqual(len(encoded_masked), len(encoded_1))
    #             mask_loc = encoded_masked.index(mask_ind)
    #             encoded_masked[mask_loc] = encoded_1[mask_loc]

    #             self.assertEqual(encoded_masked, encoded_1)

    def test_special_tokens_mask(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                # Testing single inputs
                encoded_sequence = tokenizer.encode(
                    sequence_0,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                encoded_sequence_dict = tokenizer.encode(
                    sequence_0,
                    add_special_tokens=True,
                    return_special_tokens_mask=True  # , add_prefix_space=False
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict[
                    "special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask),
                                 len(encoded_sequence_w_special))

                filtered_sequence = [
                    x for i, x in enumerate(encoded_sequence_w_special)
                    if not special_tokens_mask[i]
                ]
                self.assertEqual(encoded_sequence, filtered_sequence)

    def test_special_tokens_mask_input_pairs(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                sequence_1 = "This one too please."
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

    def test_padding_side_in_kwargs(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(
                    f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, padding_side="left", **kwargs)
                self.assertEqual(tokenizer.padding_side, "left")

                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, padding_side="right", **kwargs)
                self.assertEqual(tokenizer.padding_side, "right")

                self.assertRaises(
                    ValueError,
                    self.tokenizer_class.from_pretrained,
                    pretrained_name,
                    padding_side="unauthorized",
                    **kwargs,
                )

    def test_truncation_side_in_kwargs(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(
                    f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, truncation_side="left", **kwargs)
                self.assertEqual(tokenizer.truncation_side, "left")

                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, truncation_side="right", **kwargs)
                self.assertEqual(tokenizer.truncation_side, "right")

                self.assertRaises(
                    ValueError,
                    self.tokenizer_class.from_pretrained,
                    pretrained_name,
                    truncation_side="unauthorized",
                    **kwargs,
                )

    def test_right_and_left_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(sequence)['input_ids']
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    sequence,
                    max_length=sequence_length + padding_size,
                    padding="max_length")['input_ids']
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size,
                                 padded_sequence_length)
                self.assertEqual(
                    encoded_sequence + [padding_idx] * padding_size,
                    padded_sequence)

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "left"
                encoded_sequence = tokenizer.encode(sequence)['input_ids']
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    sequence,
                    max_length=sequence_length + padding_size,
                    padding="max_length")['input_ids']
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size,
                                 padded_sequence_length)
                self.assertEqual([padding_idx] * padding_size +
                                 encoded_sequence, padded_sequence)

                # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_padding'
                encoded_sequence = tokenizer.encode(sequence)['input_ids']
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(
                    sequence, padding=True)['input_ids']
                padded_sequence_right_length = len(padded_sequence_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertEqual(encoded_sequence, padded_sequence_right)

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(
                    sequence, padding="longest")['input_ids']
                padded_sequence_left_length = len(padded_sequence_left)
                self.assertEqual(sequence_length, padded_sequence_left_length)
                self.assertEqual(encoded_sequence, padded_sequence_left)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(sequence)['input_ids']
                padded_sequence_right_length = len(padded_sequence_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertEqual(encoded_sequence, padded_sequence_right)

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(
                    sequence, padding=False)['input_ids']
                padded_sequence_left_length = len(padded_sequence_left)
                self.assertEqual(sequence_length, padded_sequence_left_length)
                self.assertEqual(encoded_sequence, padded_sequence_left)

    def test_right_and_left_truncation(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "This is a test sequence"

                # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                truncation_size = 3
                tokenizer.truncation_side = "right"
                encoded_sequence = tokenizer.encode(
                    sequence,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                sequence_length = len(encoded_sequence)
                # Remove EOS/BOS tokens
                truncated_sequence = tokenizer.encode(
                    sequence,
                    max_length=sequence_length - truncation_size,
                    truncation=True,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                truncated_sequence_length = len(truncated_sequence)
                self.assertEqual(sequence_length,
                                 truncated_sequence_length + truncation_size)
                self.assertEqual(encoded_sequence[:-truncation_size],
                                 truncated_sequence)

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the truncation flag set to True
                tokenizer.truncation_side = "left"
                sequence_length = len(encoded_sequence)
                truncated_sequence = tokenizer.encode(
                    sequence,
                    max_length=sequence_length - truncation_size,
                    truncation=True,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                truncated_sequence_length = len(truncated_sequence)
                self.assertEqual(sequence_length,
                                 truncated_sequence_length + truncation_size)
                self.assertEqual(encoded_sequence[truncation_size:],
                                 truncated_sequence)

                # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_truncation'
                sequence_length = len(encoded_sequence)

                tokenizer.truncation_side = "right"
                truncated_sequence_right = tokenizer.encode(
                    sequence,
                    truncation=True,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                truncated_sequence_right_length = len(truncated_sequence_right)
                self.assertEqual(sequence_length,
                                 truncated_sequence_right_length)
                self.assertEqual(encoded_sequence, truncated_sequence_right)

                tokenizer.truncation_side = "left"
                truncated_sequence_left = tokenizer.encode(
                    sequence,
                    truncation="longest_first",
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                truncated_sequence_left_length = len(truncated_sequence_left)
                self.assertEqual(sequence_length,
                                 truncated_sequence_left_length)
                self.assertEqual(encoded_sequence, truncated_sequence_left)

                tokenizer.truncation_side = "right"
                truncated_sequence_right = tokenizer.encode(
                    sequence,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                truncated_sequence_right_length = len(truncated_sequence_right)
                self.assertEqual(sequence_length,
                                 truncated_sequence_right_length)
                self.assertEqual(encoded_sequence, truncated_sequence_right)

                tokenizer.truncation_side = "left"
                truncated_sequence_left = tokenizer.encode(
                    sequence,
                    truncation=False,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                truncated_sequence_left_length = len(truncated_sequence_left)
                self.assertEqual(sequence_length,
                                 truncated_sequence_left_length)
                self.assertEqual(encoded_sequence, truncated_sequence_left)

    def test_padding_to_max_length(self):
        """We keep this test for backward compatibility but it should be remove when `pad_to_max_seq_len` is deprecated."""
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(sequence)['input_ids']
                sequence_length = len(encoded_sequence)
                # FIXME: the next line should be padding(max_length) to avoid warning
                padded_sequence = tokenizer.encode(
                    sequence,
                    max_length=sequence_length + padding_size,
                    pad_to_max_seq_len=True)['input_ids']
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size,
                                 padded_sequence_length)
                self.assertEqual(
                    encoded_sequence + [padding_idx] * padding_size,
                    padded_sequence)

                # Check that nothing is done when a maximum length is not specified
                encoded_sequence = tokenizer.encode(sequence)['input_ids']
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(
                    sequence, pad_to_max_seq_len=True)['input_ids']
                padded_sequence_right_length = len(padded_sequence_right)
                self.assertEqual(sequence_length, padded_sequence_right_length)
                self.assertEqual(encoded_sequence, padded_sequence_right)

    def test_padding_to_multiple_of(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.pad_token is None:
                    self.skipTest("No padding token.")
                else:
                    empty_tokens = tokenizer("",
                                             padding=True,
                                             pad_to_multiple_of=8)
                    normal_tokens = tokenizer("This is a sample input",
                                              padding=True,
                                              pad_to_multiple_of=8)
                    for key, value in empty_tokens.items():
                        self.assertEqual(
                            len(value) % 8, 0,
                            f"BatchEncoding.{key} is not multiple of 8")
                    for key, value in normal_tokens.items():
                        self.assertEqual(
                            len(value) % 8, 0,
                            f"BatchEncoding.{key} is not multiple of 8")

                    normal_tokens = tokenizer("This", pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertNotEqual(
                            len(value) % 8, 0,
                            f"BatchEncoding.{key} is not multiple of 8")

                    # Should also work with truncation
                    normal_tokens = tokenizer("This",
                                              padding=True,
                                              truncation=True,
                                              pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertEqual(
                            len(value) % 8, 0,
                            f"BatchEncoding.{key} is not multiple of 8")

                    # truncation to something which is not a multiple of pad_to_multiple_of raises an error
                    self.assertRaises(
                        ValueError,
                        tokenizer.__call__,
                        "This",
                        padding=True,
                        truncation=True,
                        max_length=12,
                        pad_to_multiple_of=8,
                    )

    def test_padding_with_attention_mask(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.pad_token is None:
                    self.skipTest("No padding token.")
                if "attention_mask" not in tokenizer.model_input_names:
                    self.skipTest("This model does not use attention mask.")

                features = [
                    {
                        "input_ids": [1, 2, 3, 4, 5, 6],
                        "attention_mask": [1, 1, 1, 1, 1, 0]
                    },
                    {
                        "input_ids": [1, 2, 3],
                        "attention_mask": [1, 1, 0]
                    },
                ]
                padded_features = tokenizer.pad(features)
                if tokenizer.padding_side == "right":
                    self.assertListEqual(
                        padded_features["attention_mask"],
                        [[1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0]])
                else:
                    self.assertListEqual(
                        padded_features["attention_mask"],
                        [[1, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0]])

    def test_encode_plus_with_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_size = 10
                padding_idx = tokenizer.pad_token_id
                token_type_padding_idx = tokenizer.pad_token_type_id

                encoded_sequence = tokenizer.encode(
                    sequence, return_special_tokens_mask=True)
                input_ids = encoded_sequence["input_ids"]
                special_tokens_mask = encoded_sequence["special_tokens_mask"]
                sequence_length = len(input_ids)

                # Test 'longest' and 'no_padding' don't do anything
                tokenizer.padding_side = "right"

                not_padded_sequence = tokenizer.encode(
                    sequence,
                    padding=True,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence[
                    "special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                self.assertEqual(sequence_length, not_padded_sequence_length)
                self.assertEqual(input_ids, not_padded_input_ids)
                self.assertEqual(special_tokens_mask,
                                 not_padded_special_tokens_mask)

                not_padded_sequence = tokenizer.encode(
                    sequence,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence[
                    "special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                self.assertEqual(sequence_length, not_padded_sequence_length)
                self.assertEqual(input_ids, not_padded_input_ids)
                self.assertEqual(special_tokens_mask,
                                 not_padded_special_tokens_mask)

                # Test right padding
                tokenizer.padding_side = "right"

                right_padded_sequence = tokenizer.encode(
                    sequence,
                    max_length=sequence_length + padding_size,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )
                right_padded_input_ids = right_padded_sequence["input_ids"]

                right_padded_special_tokens_mask = right_padded_sequence[
                    "special_tokens_mask"]
                right_padded_sequence_length = len(right_padded_input_ids)

                self.assertEqual(sequence_length + padding_size,
                                 right_padded_sequence_length)
                self.assertEqual(input_ids + [padding_idx] * padding_size,
                                 right_padded_input_ids)
                self.assertEqual(special_tokens_mask + [1] * padding_size,
                                 right_padded_special_tokens_mask)

                # Test left padding
                tokenizer.padding_side = "left"
                left_padded_sequence = tokenizer.encode(
                    sequence,
                    max_length=sequence_length + padding_size,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )
                left_padded_input_ids = left_padded_sequence["input_ids"]
                left_padded_special_tokens_mask = left_padded_sequence[
                    "special_tokens_mask"]
                left_padded_sequence_length = len(left_padded_input_ids)

                self.assertEqual(sequence_length + padding_size,
                                 left_padded_sequence_length)
                self.assertEqual([padding_idx] * padding_size + input_ids,
                                 left_padded_input_ids)
                self.assertEqual([1] * padding_size + special_tokens_mask,
                                 left_padded_special_tokens_mask)

                if "token_type_ids" in tokenizer.model_input_names:
                    token_type_ids = encoded_sequence["token_type_ids"]
                    left_padded_token_type_ids = left_padded_sequence[
                        "token_type_ids"]
                    right_padded_token_type_ids = right_padded_sequence[
                        "token_type_ids"]

                    self.assertEqual(
                        token_type_ids +
                        [token_type_padding_idx] * padding_size,
                        right_padded_token_type_ids)
                    self.assertEqual([token_type_padding_idx] * padding_size +
                                     token_type_ids, left_padded_token_type_ids)

                if "attention_mask" in tokenizer.model_input_names:
                    attention_mask = encoded_sequence["attention_mask"]
                    right_padded_attention_mask = right_padded_sequence[
                        "attention_mask"]
                    left_padded_attention_mask = left_padded_sequence[
                        "attention_mask"]

                    self.assertEqual(attention_mask + [0] * padding_size,
                                     right_padded_attention_mask)
                    self.assertEqual([0] * padding_size + attention_mask,
                                     left_padded_attention_mask)

    def test_separate_tokenizers(self):
        # This tests that tokenizers don't impact others. Unfortunately the case where it fails is when
        # we're loading an S3 configuration from a pre-trained identifier, and we have no way of testing those today.

        tokenizers = self.get_tokenizers(random_argument=True)
        new_tokenizers = self.get_tokenizers(random_argument=False)

        for tokenizer, new_tokenizer in zip(tokenizers, new_tokenizers):
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertTrue(tokenizer.init_kwargs["random_argument"])
                self.assertTrue(tokenizer.init_kwargs["random_argument"])
                self.assertFalse(new_tokenizer.init_kwargs["random_argument"])

    def test_get_vocab(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_dict = tokenizer.get_vocab()
                self.assertIsInstance(vocab_dict, dict)
                self.assertGreaterEqual(len(tokenizer), len(vocab_dict))

                vocab = [
                    tokenizer.convert_ids_to_tokens(i)
                    for i in range(len(tokenizer))
                ]
                self.assertEqual(len(vocab), len(tokenizer))

                tokenizer.add_tokens(["asdfasdfasdfasdf"])
                vocab = [
                    tokenizer.convert_ids_to_tokens(i)
                    for i in range(len(tokenizer))
                ]
                self.assertEqual(len(vocab), len(tokenizer))

    def test_conversion_reversible(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab = tokenizer.get_vocab()
                for word, ind in vocab.items():
                    if word == tokenizer.unk_token:
                        continue
                    self.assertEqual(tokenizer.convert_tokens_to_ids(word), ind)
                    self.assertEqual(tokenizer.convert_ids_to_tokens(ind), word)

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                # Test not batched
                encoded_sequences_1 = tokenizer.encode(
                    sequences[0],
                    return_token_type_ids=False,
                    return_attention_mask=True)
                encoded_sequences_2 = tokenizer(sequences[0],
                                                return_token_type_ids=False,
                                                return_attention_mask=True)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test not batched pairs
                encoded_sequences_1 = tokenizer.encode(
                    sequences[0],
                    sequences[1],
                    return_token_type_ids=False,
                    return_attention_mask=True)
                encoded_sequences_2 = tokenizer(sequences[0],
                                                sequences[1],
                                                return_token_type_ids=False,
                                                return_attention_mask=True)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched
                encoded_sequences_1 = tokenizer.batch_encode(
                    sequences,
                    return_token_type_ids=False,
                    return_attention_mask=True)
                encoded_sequences_2 = tokenizer(sequences,
                                                return_token_type_ids=False,
                                                return_attention_mask=True)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched pairs
                encoded_sequences_1 = tokenizer.batch_encode(
                    list(zip(sequences, sequences)),
                    return_token_type_ids=False,
                    return_attention_mask=True)
                encoded_sequences_2 = tokenizer(sequences,
                                                sequences,
                                                return_token_type_ids=False,
                                                return_attention_mask=True)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

    def test_batch_encode_plus_batch_sequence_length(self):
        # Tests that all encoded values have the correct size
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                encoded_sequences = [
                    tokenizer.encode(sequence) for sequence in sequences
                ]
                encoded_sequences_batch = tokenizer.batch_encode(sequences,
                                                                 padding=False)
                self.assertListEqual(
                    encoded_sequences,
                    self.convert_batch_encode_plus_format_to_encode_plus(
                        encoded_sequences_batch))

                maximum_length = len(
                    max([
                        encoded_sequence["input_ids"]
                        for encoded_sequence in encoded_sequences
                    ],
                        key=len))

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences_padded = [
                    tokenizer.encode(sequence,
                                     max_length=maximum_length,
                                     padding="max_length")
                    for sequence in sequences
                ]

                encoded_sequences_batch_padded = tokenizer.batch_encode(
                    sequences, padding=True)
                self.assertListEqual(
                    encoded_sequences_padded,
                    self.convert_batch_encode_plus_format_to_encode_plus(
                        encoded_sequences_batch_padded),
                )

                # check 'longest' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode(
                    sequences, padding=True)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode(
                    sequences,
                    max_length=maximum_length + 10,
                    padding="longest")
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

                # check 'no_padding' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode(
                    sequences, padding=False)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode(
                    sequences, max_length=maximum_length + 10, padding=False)
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

    def test_added_token_are_matched_longest_first(self):
        tokenizers = self.get_tokenizers(fast=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                try:
                    tokenizer.add_tokens([AddedToken("extra_id_1")])
                    tokenizer.add_tokens([AddedToken("extra_id_100")])
                except Exception:
                    # Canine cannot add tokens which are not codepoints
                    self.skipTest("Cannot add those Added tokens")

                # XXX: This used to split on `extra_id_1` first we're matching
                # longest first now.
                tokens = tokenizer.tokenize("This is some extra_id_100")
                self.assertIn("extra_id_100", tokens)

        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokenizer.add_tokens([AddedToken("extra_id_100")])
                tokenizer.add_tokens([AddedToken("extra_id_1")])

                tokens = tokenizer.tokenize("This is some extra_id_100")
                self.assertIn("extra_id_100", tokens)

    def test_added_token_serializable(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                new_token = AddedToken("new_token", lstrip=True)
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": [new_token]})

                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    tokenizer.save_pretrained(tmp_dir_name)
                    tokenizer.from_pretrained(tmp_dir_name)

    def test_batch_encode_plus_padding(self):
        # Test that padded sequences are equivalent between batch_encode_plus and encode_plus

        # Right padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences = [
                    tokenizer.encode(sequence,
                                     max_length=max_length,
                                     padding="max_length")
                    for sequence in sequences
                ]
                encoded_sequences_batch = tokenizer.batch_encode(
                    sequences, max_length=max_length, padding="max_length")
                self.assertListEqual(
                    encoded_sequences,
                    self.convert_batch_encode_plus_format_to_encode_plus(
                        encoded_sequences_batch))

        # Left padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokenizer.padding_side = "left"
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences = [
                    tokenizer.encode(sequence,
                                     max_length=max_length,
                                     padding="max_length")
                    for sequence in sequences
                ]
                encoded_sequences_batch = tokenizer.batch_encode(
                    sequences, max_length=max_length, padding="max_length")
                self.assertListEqual(
                    encoded_sequences,
                    self.convert_batch_encode_plus_format_to_encode_plus(
                        encoded_sequences_batch))

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
                # sequence = " " + sequence  # To be sure the byte-level tokenizers are feeling good
                token_sequence = sequence.split()
                # sequence_no_prefix_space = sequence.strip()

                # Test encode for pretokenized inputs
                output = tokenizer.encode(token_sequence,
                                          is_split_into_words=True,
                                          return_token_type_ids=None,
                                          add_special_tokens=False)['input_ids']
                output_sequence = tokenizer.encode(
                    sequence,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                self.assertEqual(output, output_sequence)

                output = tokenizer.encode(token_sequence,
                                          is_split_into_words=True,
                                          add_special_tokens=True)['input_ids']
                output_sequence = tokenizer.encode(
                    sequence, add_special_tokens=True)['input_ids']
                self.assertEqual(output, output_sequence)

                # Test encode_plus for pretokenized inputs
                output = tokenizer.encode(token_sequence,
                                          is_split_into_words=True,
                                          return_token_type_ids=None,
                                          add_special_tokens=False)
                output_sequence = tokenizer.encode(sequence,
                                                   return_token_type_ids=None,
                                                   add_special_tokens=False)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.encode(token_sequence,
                                          is_split_into_words=True,
                                          add_special_tokens=True)
                output_sequence = tokenizer.encode(sequence,
                                                   add_special_tokens=True)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

                # Test batch_encode_plus for pretokenized inputs
                sequence_batch = [sequence.strip()] * 2 + [
                    sequence.strip() + " " + sequence.strip()
                ]
                token_sequence_batch = [s.split() for s in sequence_batch]
                sequence_batch_cleaned_up_spaces = [
                    " " + " ".join(s) for s in token_sequence_batch
                ]

                output = tokenizer.batch_encode(token_sequence_batch,
                                                is_split_into_words=True,
                                                return_token_type_ids=None,
                                                add_special_tokens=False)
                output_sequence = tokenizer.batch_encode(
                    sequence_batch_cleaned_up_spaces,
                    return_token_type_ids=None,
                    add_special_tokens=False)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.batch_encode(token_sequence_batch,
                                                is_split_into_words=True,
                                                add_special_tokens=True)
                output_sequence = tokenizer.batch_encode(
                    sequence_batch_cleaned_up_spaces, add_special_tokens=True)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

                # Test encode for pretokenized inputs pairs
                output = tokenizer.encode(token_sequence,
                                          token_sequence,
                                          is_split_into_words=True,
                                          return_token_type_ids=None,
                                          add_special_tokens=False)['input_ids']
                output_sequence = tokenizer.encode(
                    sequence,
                    sequence,
                    return_token_type_ids=None,
                    add_special_tokens=False)['input_ids']
                self.assertEqual(output, output_sequence)
                output = tokenizer.encode(token_sequence,
                                          token_sequence,
                                          is_split_into_words=True,
                                          add_special_tokens=True)['input_ids']
                output_sequence = tokenizer.encode(
                    sequence, sequence, add_special_tokens=True)['input_ids']
                self.assertEqual(output, output_sequence)

                # Test encode_plus for pretokenized inputs pairs
                output = tokenizer.encode(token_sequence,
                                          token_sequence,
                                          is_split_into_words=True,
                                          return_token_type_ids=None,
                                          add_special_tokens=False)
                output_sequence = tokenizer.encode(sequence,
                                                   sequence,
                                                   return_token_type_ids=None,
                                                   add_special_tokens=False)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.encode(token_sequence,
                                          token_sequence,
                                          is_split_into_words=True,
                                          add_special_tokens=True)
                output_sequence = tokenizer.encode(sequence,
                                                   sequence,
                                                   add_special_tokens=True)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

                # Test batch_encode_plus for pretokenized inputs pairs
                sequence_pair_batch = [
                    (sequence.strip(), sequence.strip())
                ] * 2 + [(sequence.strip() + " " + sequence.strip(),
                          sequence.strip())]
                token_sequence_pair_batch = [
                    tuple(s.split() for s in pair)
                    for pair in sequence_pair_batch
                ]
                sequence_pair_batch_cleaned_up_spaces = [
                    tuple(" " + " ".join(s) for s in pair)
                    for pair in token_sequence_pair_batch
                ]

                output = tokenizer.batch_encode(token_sequence_pair_batch,
                                                is_split_into_words=True,
                                                return_token_type_ids=None,
                                                add_special_tokens=False)
                output_sequence = tokenizer.batch_encode(
                    sequence_pair_batch_cleaned_up_spaces,
                    return_token_type_ids=None,
                    add_special_tokens=False)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.batch_encode(token_sequence_pair_batch,
                                                is_split_into_words=True,
                                                add_special_tokens=True)
                output_sequence = tokenizer.batch_encode(
                    sequence_pair_batch_cleaned_up_spaces,
                    add_special_tokens=True)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

    def test_prepare_for_model(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                string_sequence = "Testing the prepare_for_model method."
                ids = tokenizer.encode(string_sequence,
                                       return_token_type_ids=None,
                                       add_special_tokens=False)['input_ids']
                prepared_input_dict = tokenizer.prepare_for_model(
                    ids, add_special_tokens=True)

                input_dict = tokenizer.encode(string_sequence,
                                              add_special_tokens=True)

                self.assertEqual(input_dict, prepared_input_dict)

    def test_batch_encode_plus_overflowing_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            string_sequences = ["Testing the prepare_for_model method.", "Test"]

            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            tokenizer.batch_encode(string_sequences,
                                   return_overflowing_tokens=True,
                                   truncation=True,
                                   padding=True,
                                   max_length=3)

    def _check_no_pad_token_padding(self, tokenizer, sequences):
        # if tokenizer does not have pad_token_id, an error should be thrown
        if tokenizer.pad_token_id is None:
            with self.assertRaises(ValueError):
                if isinstance(sequences, list):
                    tokenizer.batch_encode(sequences, padding="longest")
                else:
                    tokenizer.encode(sequences, padding=True)

            # add pad_token_id to pass subsequent tests
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    def check_subword_sampling(
        self,
        tokenizer: PretrainedTokenizer,
        text: str = None,
    ) -> None:
        """
        Check if the tokenizer generates different results when subword regularization is enabled.

        Subword regularization augments training data with subword sampling.
        This has a random component.

        Args:
            tokenizer: The tokenizer to check.
            text: The text to use for the checks.
        """
        text = "This is a test for subword regularization." if text is None else text
        if self.test_sentencepiece_ignore_case:
            text = text.lower()

        tokens_list = []
        for _ in range(5):
            tokens_list.append(tokenizer.tokenize(text))

        # the list of different pairs of tokens_list
        combinations = itertools.combinations(tokens_list, 2)

        # check of sampling is done
        subword_sampling_found = False
        for combination in combinations:
            if combination[0] != combination[1]:
                subword_sampling_found = True
        self.assertTrue(subword_sampling_found)

        # check if converting back to original text works
        for tokens in tokens_list:
            if self.test_sentencepiece_ignore_case:
                self.assertEqual(
                    text,
                    tokenizer.convert_tokens_to_string(tokens).lower())
            else:
                self.assertEqual(text,
                                 tokenizer.convert_tokens_to_string(tokens))

    def test_add_tokens(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(
                    f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, **kwargs)

                vocab_size = len(tokenizer)
                self.assertEqual(tokenizer.add_tokens(""), 0)
                self.assertEqual(tokenizer.add_tokens("testoken"), 1)
                self.assertEqual(
                    tokenizer.add_tokens(["testoken1", "testtoken2"]), 2)
                self.assertEqual(len(tokenizer), vocab_size + 3)

                self.assertEqual(tokenizer.add_special_tokens({}), 0)
                self.assertEqual(
                    tokenizer.add_special_tokens({
                        "bos_token": "[BOS]",
                        "eos_token": "[EOS]"
                    }), 2)
                self.assertRaises(AssertionError, tokenizer.add_special_tokens,
                                  {"additional_special_tokens": "<testtoken1>"})
                self.assertEqual(
                    tokenizer.add_special_tokens(
                        {"additional_special_tokens": ["<testtoken2>"]}), 1)
                self.assertEqual(
                    tokenizer.add_special_tokens({
                        "additional_special_tokens":
                        ["<testtoken3>", "<testtoken4>"]
                    }), 2)
                self.assertIn(
                    "<testtoken3>",
                    tokenizer.special_tokens_map["additional_special_tokens"])
                self.assertIsInstance(
                    tokenizer.special_tokens_map["additional_special_tokens"],
                    list)
                self.assertGreaterEqual(
                    len(tokenizer.
                        special_tokens_map["additional_special_tokens"]), 2)

                self.assertEqual(len(tokenizer), vocab_size + 8)

    def test_offsets_mapping(self):
        if not self.test_offsets:
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(
                    f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                tokenizer = self.tokenizer_class.from_pretrained(
                    pretrained_name, **kwargs)

                text = "Wonderful no inspiration example with subtoken"
                pair = "Along with an awesome pair"

                # No pair
                tokens_with_offsets = tokenizer.encode(
                    text,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=True,
                    add_special_tokens=True)
                added_tokens = tokenizer.num_special_tokens_to_add(False)
                offsets = tokens_with_offsets["offset_mapping"]

                # Assert there is the same number of tokens and offsets
                self.assertEqual(len(offsets),
                                 len(tokens_with_offsets["input_ids"]))

                # Assert there is online added_tokens special_tokens
                self.assertEqual(
                    sum(tokens_with_offsets["special_tokens_mask"]),
                    added_tokens)

                # Pairs
                tokens_with_offsets = tokenizer.encode(
                    text,
                    pair,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=True,
                    add_special_tokens=True)
                added_tokens = tokenizer.num_special_tokens_to_add(True)
                offsets = tokens_with_offsets["offset_mapping"]

                # Assert there is the same number of tokens and offsets
                self.assertEqual(len(offsets),
                                 len(tokens_with_offsets["input_ids"]))

                # Assert there is online added_tokens special_tokens
                self.assertEqual(
                    sum(tokens_with_offsets["special_tokens_mask"]),
                    added_tokens)

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(
            self):
        tokenizer_list = [(self.tokenizer_class, self.get_tokenizer())]

        for tokenizer_class, tokenizer_utils in tokenizer_list:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer_utils.save_pretrained(tmp_dir)

                with open(os.path.join(tmp_dir, "special_tokens_map.json"),
                          encoding="utf-8") as json_file:
                    special_tokens_map = json.load(json_file)

                with open(os.path.join(tmp_dir, "tokenizer_config.json"),
                          encoding="utf-8") as json_file:
                    tokenizer_config = json.load(json_file)

                special_tokens_map["additional_special_tokens"] = [
                    "an_additional_special_token"
                ]
                tokenizer_config["additional_special_tokens"] = [
                    "an_additional_special_token"
                ]

                with open(os.path.join(tmp_dir, "special_tokens_map.json"),
                          "w",
                          encoding="utf-8") as outfile:
                    json.dump(special_tokens_map, outfile)
                with open(os.path.join(tmp_dir, "tokenizer_config.json"),
                          "w",
                          encoding="utf-8") as outfile:
                    json.dump(tokenizer_config, outfile)

                # the following checks allow us to verify that our test works as expected, i.e. that the tokenizer takes
                # into account the new value of additional_special_tokens given in the "tokenizer_config.json" and
                # "special_tokens_map.json" files
                tokenizer_without_change_in_init = tokenizer_class.from_pretrained(
                    tmp_dir, )
                self.assertIn(
                    "an_additional_special_token",
                    tokenizer_without_change_in_init.additional_special_tokens)

                self.assertIn("an_additional_special_token",
                              tokenizer_without_change_in_init.get_vocab())
                self.assertEqual(
                    ["an_additional_special_token"],
                    tokenizer_without_change_in_init.convert_ids_to_tokens(
                        tokenizer_without_change_in_init.convert_tokens_to_ids(
                            ["an_additional_special_token"])),
                )

                # Now we test that we can change the value of additional_special_tokens in the from_pretrained
                new_added_tokens = [
                    AddedToken("a_new_additional_special_token", lstrip=True)
                ]
                tokenizer = tokenizer_class.from_pretrained(
                    tmp_dir,
                    additional_special_tokens=new_added_tokens,
                )

                self.assertIn("a_new_additional_special_token",
                              tokenizer.additional_special_tokens)
                self.assertEqual(
                    ["a_new_additional_special_token"],
                    tokenizer.convert_ids_to_tokens(
                        tokenizer.convert_tokens_to_ids(
                            ["a_new_additional_special_token"])),
                )

    def test_convert_tokens_to_string_format(self):
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokens = ["this", "is", "a", "test"]
                string = tokenizer.convert_tokens_to_string(tokens)

                self.assertIsInstance(string, str)


class TrieTest(unittest.TestCase):

    def test_trie(self):
        trie = Trie()
        trie.add("Hello 友達")
        self.assertEqual(
            trie.data,
            {"H": {
                "e": {
                    "l": {
                        "l": {
                            "o": {
                                " ": {
                                    "友": {
                                        "達": {
                                            "": 1
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }})
        trie.add("Hello")
        trie.data
        self.assertEqual(trie.data, {
            "H": {
                "e": {
                    "l": {
                        "l": {
                            "o": {
                                "": 1,
                                " ": {
                                    "友": {
                                        "達": {
                                            "": 1
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        })

    def test_trie_split(self):
        trie = Trie()
        self.assertEqual(trie.split("[CLS] This is a extra_id_100"),
                         ["[CLS] This is a extra_id_100"])
        trie.add("[CLS]")
        trie.add("extra_id_1")
        trie.add("extra_id_100")
        self.assertEqual(trie.split("[CLS] This is a extra_id_100"),
                         ["[CLS]", " This is a ", "extra_id_100"])

    def test_trie_single(self):
        trie = Trie()
        trie.add("A")
        self.assertEqual(trie.split("ABC"), ["A", "BC"])
        self.assertEqual(trie.split("BCA"), ["BC", "A"])

    def test_trie_final(self):
        trie = Trie()
        trie.add("TOKEN]")
        trie.add("[SPECIAL_TOKEN]")
        self.assertEqual(trie.split("This is something [SPECIAL_TOKEN]"),
                         ["This is something ", "[SPECIAL_TOKEN]"])

    def test_trie_subtokens(self):
        trie = Trie()
        trie.add("A")
        trie.add("P")
        trie.add("[SPECIAL_TOKEN]")
        self.assertEqual(trie.split("This is something [SPECIAL_TOKEN]"),
                         ["This is something ", "[SPECIAL_TOKEN]"])

    def test_trie_suffix_tokens(self):
        trie = Trie()
        trie.add("AB")
        trie.add("B")
        trie.add("C")
        self.assertEqual(trie.split("ABC"), ["AB", "C"])

    def test_trie_skip(self):
        trie = Trie()
        trie.add("ABC")
        trie.add("B")
        trie.add("CD")
        self.assertEqual(trie.split("ABCD"), ["ABC", "D"])

    def test_cut_text_hardening(self):
        # Even if the offsets are wrong, we necessarily output correct string
        # parts.
        trie = Trie()
        parts = trie.cut_text("ABC", [0, 0, 2, 1, 2, 3])
        self.assertEqual(parts, ["AB", "C"])
