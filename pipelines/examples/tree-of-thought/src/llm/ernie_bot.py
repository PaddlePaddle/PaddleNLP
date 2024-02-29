# coding=utf8, ErnestinaQiu

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import re
import time

import erniebot

Ernie_llm_list = ["ernie-3.5", "ernie-4.0"]


class Ernie:
    def __init__(self, model):
        self.query = []
        self.query_count = 0
        self.max_prompt_len = 512

    def create(self, model, messages, temperature=0.6):
        # :input @messages is like [{'role': 'user', 'content': "请问你能以《你好，世界》为题，写一首现代诗吗？"}]
        # :output @out is a string
        self.query.append(messages[0])
        self.query_count += len(messages[0]["content"])

        while self.query_count > self.max_prompt_len and len(self.query) > 2:
            _pop = self.query.pop(0)
            _pop_len = len(_pop["content"])
            self.query_count -= _pop_len
            _pop = self.query.pop(0)
            _pop_len = len(_pop["content"])
            self.query_count -= _pop_len

        request_success = False
        while not request_success:
            try:
                resp = erniebot.ChatCompletion.create(
                    model=model,
                    messages=self.query,
                    system="""你是一个任务型助手，你需要解决数学问题，你需要严格遵守以下要求：
                            1.只能用英文、数字、数学符号和标点符号进行回复。
                            3.严格按照用户的指令进行回复。
                            4.涉及到计算过程只能使用数学表达式回复
                            """,
                )
                request_success = True
            except:
                time.sleep(60)
                continue
        out = resp.to_message()["content"]
        eles = out.split("\n")
        for i in range(len(eles)):
            sentence = eles[i]
            if contains_chinese(sentence) and not contains_number(sentence):
                continue
            if contains_number(sentence) and contains_math_symbols(sentence):
                break
            if contains_english(sentence):
                break
        new_out = "\n".join(eles[i:])
        self.query.append(resp.to_message())
        return new_out


def contains_number(input_string):
    # 检查字符串中是否存在中文 和 数字
    return bool(re.search(r"\d", input_string))


def contains_chinese(input_string):
    return bool(re.search(r"[\u4e00-\u9fff]", input_string))


def contains_english(input_string):
    return bool(re.search(r"[a-zA-Z]", input_string))


def contains_math_symbols(input_string):
    # 这里我们对特殊字符进行了转义，因为它们在正则表达式中有特殊含义
    return bool(re.search(r"[\+\-\*/]", input_string))
