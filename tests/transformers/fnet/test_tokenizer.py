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
import unittest

from paddlenlp.transformers import FNetTokenizer

from ...testing_utils import get_tests_dir
from ..test_tokenizer_common import TokenizerTesterMixin

SAMPLE_VOCAB = get_tests_dir("fixtures/spiece.model")


class TestTokenizationFNet(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = FNetTokenizer
    from_pretrained_vocab_key = "sentencepiece_model_file"
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True

    def setUp(self):
        super().setUp()
        # We have a SentencePiece fixture for testing
        tokenizer = FNetTokenizer(SAMPLE_VOCAB)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def test_add_special_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, ids = self.get_clean_sequence(tokenizer)

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
