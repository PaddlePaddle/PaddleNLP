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

import unittest

from paddlenlp.transformers.chatglm_v2.tokenizer import ChatGLMv2Tokenizer
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
}


class ChatGLMv2TokenizationTest(unittest.TestCase):

    tokenizer_class = ChatGLMv2Tokenizer
    test_decode_token = True

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        tokenizer = ChatGLMv2Tokenizer.from_pretrained("THUDM/chatglm2-6b", **kwargs)
        return tokenizer

    def test_encode_special_tokens(self):
        tokenizer = self.get_tokenizer()

        query = "[gMASK]</s>"
        tokens = tokenizer.tokenize(query)
        self.assertEqual(len(tokens), 2)

        outputs = tokenizer.encode(query, add_special_tokens=False)
        self.assertEqual(len(outputs["input_ids"]), 2)


class ChatGLMv3TokenizationTest(unittest.TestCase):
    tokenizer_class = ChatGLMv2Tokenizer

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        return ChatGLMv2Tokenizer.from_pretrained("THUDM/chatglm3-6b", **kwargs)

    def test_encode_special_tokens(self):
        tokenizer = self.get_tokenizer()

        query = "[gMASK]<|user|><|assistant|></s>"
        tokens = tokenizer.tokenize(query)
        self.assertEqual(len(tokens), 4)
