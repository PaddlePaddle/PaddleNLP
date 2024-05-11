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

from function_call.schema import (
    FunctionCallMessage,
    Message,
    group_round_messages,
    parse_messages,
)
from tqdm import tqdm


def process_raw_message(raw_message: dict):
    tools = []
    openapi_dict = json.loads(raw_message["Documentation"])
    for path, path_info in openapi_dict["paths"].items():
        path = path.strip("/")
        path_info = list(path_info.values())[0]
        tools.append(
            dict(
                name=path,
                description=path_info.get("description", path_info.get("summary", path)),
                parameters=path_info.get("parameters", {}),
            )
        )

    # construct message
    messages = []
    for round_messages in raw_message["Instances"]:
        if "input" not in round_messages:
            continue
        messages.append(Message(role="user", content=round_messages["input"]))
        for inter_index, inter_step in enumerate(round_messages["intermediate_steps"]):
            action_name, _, all_info = inter_step[0]
            splits = all_info.split(f"\nAction: {action_name}\n")
            thoughts = splits[0].strip()
            action_input = all_info.split("\nAction Input: ")[-1].strip()
            messages.append(
                FunctionCallMessage(
                    role="assistant", function_call=dict(name=action_name, parameters=action_input, thoughts=thoughts)
                )
            )
            messages.append(Message(role="function", content=inter_step[-1]))
        messages.append(
            Message(
                role="assistant",
                content=f"Thought: {round_messages['Final Thought']}\nFinal Answer: {round_messages['output']}",
            )
        )
    return {"tools": tools, "messages": [m.model_dump() for m in messages]}


def convert(file: str, target_file: str) -> list[Message]:
    # 0. load files
    all_data = []
    with open(file, "r", encoding="utf-8") as f:
        raw_messages = json.load(f)
        for raw_message in tqdm(raw_messages):
            data = process_raw_message(raw_message)
            all_data.append(data)

    with open(target_file, "w+", encoding="utf-8") as f:
        for data in tqdm(all_data):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
