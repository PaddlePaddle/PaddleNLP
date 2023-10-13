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

from paddlenlp.transformers import AutoTokenizer


class ChatTemplateIntegrationTest(unittest.TestCase):
    def test_llama2_chat_template(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
        query = "who are you?"
        final_query = tokenizer.apply_chat_template(query, tokenize=False)
        expected_query = f"<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n[INST] {query} [/INST] "
        self.assertEqual(final_query, expected_query)
