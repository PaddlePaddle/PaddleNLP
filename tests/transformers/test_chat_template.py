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

import sys
import unittest
from typing import Optional

from parameterized import parameterized_class

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.transformers.tokenizer_utils import ChatTemplate


class ChatTemplateTest(unittest.TestCase):
    chat_template_config_file = "./tests/fixtures/chat_template.json"

    @property
    def chat_template(self):
        return ChatTemplate.from_file(self.chat_template_config_file)

    def test_inference_template(self):
        query = "你好"
        final_query = self.chat_template(query)
        expected_query = f"你是一个人工智能助手\nHuman: {query}<sep> Bot:"
        self.assertEqual(final_query, expected_query)

    def test_inference_conversation_template(self):
        conversations = [["你好", "您好，我是个人人工智能助手，请问有什么可以帮您。"], ["今天的天气怎么样？"]]
        final_query = self.chat_template(conversations)
        expected_query = "你是一个人工智能助手\nHuman: 你好<sep> Bot:您好，我是个人人工智能助手，请问有什么可以帮您。\nHuman: 今天的天气怎么样？<sep> Bot:"
        self.assertEqual(final_query, expected_query)

    def test_inference_conversation_template_with_one_part(self):
        conversations = [["你好"], ["今天的天气怎么样？"]]
        with self.assertRaises(AssertionError):
            self.chat_template(conversations)

    def test_null_chat_template(self):
        chat_template = ChatTemplate()
        query = "今天吃啥"
        final_query = chat_template(query)
        assert final_query == query

    def test_system_query(self):
        system = "你是一个人工智能助手:"
        query_template = "Human: {{query}}"
        chat_template = ChatTemplate(system=system, query=query_template)
        query = "今天吃啥"
        final_query = chat_template(query)
        assert final_query == system + query_template.replace("{{query}}", query)

    def test_conversation(self):
        conversation = ["Human: {{user}}<sep>", "Bot: {{bot}}\n\n"]
        chat_template = ChatTemplate(conversation=conversation)

        query = "今天吃啥"
        final_query = chat_template(query)
        assert final_query == query

        second_query = [["你好", "您好，我是个人人工智能助手"], [query]]
        final_query = chat_template(second_query)
        assert final_query == "Human: 你好<sep>Bot: 您好，我是个人人工智能助手\n\n" + query


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
        expected_query = f"<s>### Instruction:{query}  ### Response:"
        self.assertEqual(final_query, expected_query)

        # test multi turns conversation
        query = [["你好", "您好，我是个人人工智能助手"], ["今天吃啥"]]
        final_query = tokenizer.apply_chat_template(query, tokenize=False)
        expected_query = "<s>### Instruction: 你好  ### Response:您好，我是个人人工智能助手 </s>### Instruction:今天吃啥  ### Response:"
        self.assertEqual(final_query, expected_query)

    def test_chatglm_bellegroup(self):
        # refer to: https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L1267
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-v1.1")
        query = [["你好", "您好，我是个人人工智能助手"], ["今天吃啥"]]
        final_query = tokenizer.apply_chat_template(query, tokenize=False)
        expected_query = "[Round 0]\n问：你好\n答：您好，我是个人人工智能助手\n[Round 1]\n问：今天吃啥\n答：[gMASK]<sop>"
        self.assertEqual(final_query, expected_query)

    def test_bloom_bellegroup(self):
        # refer to: https://huggingface.co/BelleGroup/BELLE-7B-2M#use-model
        tokenizer = AutoTokenizer.from_pretrained("bellegroup/belle-7b-2m")
        query = "你好"
        final_query = tokenizer.apply_chat_template(query, tokenize=False)
        expected_query = f"Human: {query}\n\nAssistant:"
        self.assertEqual(final_query, expected_query)

    def test_qwen_14b_chat(self):
        # refer to: https://huggingface.co/Qwen/Qwen-14B-Chat/blob/main/qwen_generation_utils.py#L119

        # 1. test render base on query & conversation data
        tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-14b-chat")
        query = "你好"
        final_query = tokenizer.apply_chat_template(query, tokenize=False)

        expected_query = "You are a helpful assistant.\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(final_query, expected_query)

        query = [["你好", "您好，我是个人人工智能助手"], ["今天吃啥"]]
        final_query = tokenizer.apply_chat_template(query, tokenize=False)

        expected_query = (
            "You are a helpful assistant.\n<|im_start|>user\n你好<|im_end|>"
            "\n<|im_start|>assistant\n您好，我是个人人工智能助手<|im_end|>"
            "\n<|im_start|>user\n今天吃啥<|im_end|>\n<|im_start|>assistant\n"
        )
        self.assertEqual(final_query, expected_query)

        # 2. check the bos_token_id and eos_token_id
        self.assertEqual(
            tokenizer.convert_tokens_to_ids(["<|im_start|>"])[0],
            tokenizer.chat_template_bos_token_id,
        )
        self.assertEqual(
            tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0],
            tokenizer.chat_template_eos_token_id,
        )
