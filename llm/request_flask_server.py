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

import json

import requests


def send_request(query, history=None):
    data = {
        "context": query,
        "history": history,
        "top_k": 0,
        "top_p": 0.7,  # 0.0 为 greedy_search
        "temperature": 0.95,
        "repetition_penalty": 1.3,
        "max_length": 100,
        "src_length": 100,
        "min_length": 1,
    }
    res = requests.post("http://127.0.0.1:8010/api/chat", json=data, stream=True)
    text = ""
    for line in res.iter_lines():
        result = json.loads(line)

        if result["error_code"] != 0:
            text = "error-response"
            break

        result = json.loads(line)
        bot_response = result["result"]["response"]

        if bot_response["utterance"].endswith("[END]"):
            bot_response["utterance"] = bot_response["utterance"][:-5]
        text += bot_response["utterance"]

    print("result -> ", text)
    return text


send_request("你好啊")
send_request("再加一等于多少", ["一加一等于多少", "一加一等于二"])
send_request("再加一等于多少", [{"utterance": "一加一等于多少"}, {"utterance": "一加一等于二"}])
