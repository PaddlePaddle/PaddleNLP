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
from __future__ import annotations

import unittest

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.transformers.tokenizer_utils import ChatTemplate


class ChatTemplateTest(unittest.TestCase):
    chat_template_config_file = "./tests/fixtures/chat_template.json"

    def setUp(self) -> None:
        self.chat_template = ChatTemplate.from_file(self.chat_template_config_file)

    def test_inference_template(self):
        query = "你好"
        final_query = self.chat_template(query)
        expected_query = f"你是一个人工智能助手\nHuman: {query}<sep> Bot:"
        self.assertEqual(final_query, expected_query)

    def test_inference_conversation_template(self):
        conversations = [["你好", "您好，我是个人人工智能助手，请问有什么可以帮您。"], ["今天的天气怎么样？"]]
        final_query = self.chat_template(conversations)
        expected_query = "你是一个人工智能助手\nHuman: 你好<sep> Bot: 您好，我是个人人工智能助手，请问有什么可以帮您。\nHuman: 今天的天气怎么样？<sep> Bot:"
        self.assertEqual(final_query, expected_query)


class ChatTemplateIntegrationTest(unittest.TestCase):
    def test_llama2_chat_template(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
        query = "who are you?"
        final_query = tokenizer.apply_chat_template(query, tokenize=False)
        expected_query = f"<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n[INST] {query} [/INST] "
        self.assertEqual(final_query, expected_query)

    def test_linlyai_chinese_llama_2_chat_template(self):
        tokenizer = AutoTokenizer.from_pretrained("linly-ai/chinese-llama-2-7b")
        query = "你好"
        final_query = tokenizer.apply_chat_template(query, tokenize=False)
        expected_query = f"### Instruction:{query}  ### Response:"
        self.assertEqual(final_query, expected_query)
