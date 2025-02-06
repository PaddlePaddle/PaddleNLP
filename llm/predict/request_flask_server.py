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


def build_messages(query, history=None):
    """
    根据传入的 query 和 history 构造符合 OpenAI 格式的消息列表。
    如果 history 为 list 且每项为 dict，则直接使用；如果为 list 且每项为字符串，
    则依次按用户（user）与助手（assistant）交替添加；否则直接只添加当前用户消息。
    """
    messages = []
    if history:
        if isinstance(history, list):
            if all(isinstance(item, dict) for item in history):
                messages.extend(history)
            else:
                # 假设 history 按顺序依次为用户、助手、用户、助手……
                for idx, item in enumerate(history):
                    role = "user" if idx % 2 == 0 else "assistant"
                    messages.append({"role": role, "content": str(item)})
        else:
            messages.append({"role": "user", "content": str(history)})
    # 当前请求作为最新的用户消息
    messages.append({"role": "user", "content": query})
    return messages


def send_request(query, history=None):
    # 构造 OpenAI 格式的请求体
    payload = {
        "messages": build_messages(query, history),
        # 以下生成参数可根据需要调整
        # "top_k": 0,
        # "top_p": 0.7,
        # "temperature": 0.8,
        # "repetition_penalty": 1.3,
        "max_length": 100,
        "src_length": 100,
        "min_length": 1,
        # 如有需要，可增加 model 字段
        # "model": "custom-model"
    }
    # 发送 POST 请求到服务端（地址与端口应与服务端保持一致）
    res = requests.post("http://127.0.0.1:8011/v1/chat/completions", json=payload, stream=True)
    result_text = ""
    for line in res.iter_lines():
        if not line:
            continue
        try:
            response_data = json.loads(line)
        except Exception as e:
            print("解析响应出错:", e)
            continue

        # 如果返回中存在 error 字段，则直接中断并返回错误提示
        if "error" in response_data:
            error_message = response_data["error"].get("message", "Unknown error")
            result_text = f"error-response: {error_message}"
            break

        # 按 OpenAI 标准格式，结果在 choices 列表中，每个 choice 内有 message 对象
        choices = response_data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            # 如果返回文本以特定结束标志结束，则可以进行截断处理
            if content.endswith("[END]"):
                content = content[:-5]
            result_text += content

    print("result ->", result_text)
    return result_text


if __name__ == "__main__":
    # 示例调用：仅发送当前用户消息
    send_request("你好啊")
    # 示例调用：使用 history 为字符串列表（交替为用户与助手的对话）
    send_request("再加一等于多少", ["一加一等于多少", "一加一等于二"])
    # 示例调用：history 为字典格式，明确指定对话角色
    send_request("再加一等于多少", [{"role": "user", "content": "一加一等于多少"}, {"role": "assistant", "content": "一加一等于二"}])
