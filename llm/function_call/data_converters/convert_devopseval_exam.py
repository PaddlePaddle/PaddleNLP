# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import json
import sys

from function_call.schema import Message, group_round_messages, parse_messages
from tqdm import tqdm


def fill_thought_in_messages(messages: list[dict]):
    """fill thoughts to function_call field"""
    result = []
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and "function_call" in message:
            # hardcode thought to trianing dataset
            name = message["function_call"]["name"]
            thought = f"我将使用{name}工具来尝试解决此问题"
            message["function_call"]["thoughts"] = thought
        result.append(message)
    return result


def convert(file: str, target_file: str) -> list[Message]:
    # 0. load files
    all_data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            data["messages"] = data.pop("chatrounds")
            data["messages"] = fill_thought_in_messages(data["messages"])
            data["tools"] = data.pop("functions")
            all_data.append(data)

    with open(target_file, "w+", encoding="utf-8") as f:
        for data in tqdm(all_data):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
