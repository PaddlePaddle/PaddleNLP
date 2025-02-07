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
        "max_length": 2048,
        "src_length": 1024,
        "min_length": 1,
    }

    res = requests.post("http://127.0.0.1:8011/v1/chat/completions", json=payload, stream=True)
    result_text = ""
    for line in res.iter_lines():
        # https://github.com/vllm-project/vllm/blob/433c4a49230a470f13657f06e7612cde86e4fb40/examples/online_serving/openai_chat_completion_with_reasoning_streaming.py#L67-L69
        if not line:
            continue

        decoded_line = line.decode("utf-8").strip()
        # print(decoded_line)
        # OpenAI 流返回每行以 "data:" 开头
        if decoded_line.startswith("data:"):
            data = decoded_line[5:].strip()  # Remove "data:" prefix
            if data == "[DONE]":  # End of stream
                print("\nclient: Stream completed.")
                break
            try:
                response_data = json.loads(data)
                if "error" in response_data:
                    error_message = response_data["error"].get("message", "Unknown error")
                    result_text = f"error-response: {error_message}"
                    break

                # According to the OpenAI standard format, the result is in the choices list, and each choice has a message object
                choices = response_data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    result_text += content

            except Exception as e:
                print("解析响应出错:", e)
                continue

    print("result ->", result_text)
    return result_text


if __name__ == "__main__":
    # 示例调用：仅发送当前用户消息
    send_request("背诵 水调歌头")
    send_request("你好啊")
    # 示例调用：使用 history 为字符串列表（交替为用户与助手的对话）
    send_request("再加一等于多少", ["一加一等于多少", "一加一等于二"])
    # 示例调用：history 为字典格式，明确指定对话角色
    send_request("再加一等于多少", [{"role": "user", "content": "一加一等于多少"}, {"role": "assistant", "content": "一加一等于二"}])
