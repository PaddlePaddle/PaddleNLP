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
from typing import Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: Optional[str] = None

    @property
    def is_function_call_message(self) -> bool:
        return self.role == "assistant" and "function_call" in self.model_fields

    @staticmethod
    def from_dict(**message: dict) -> Message:
        if message["role"] == "assistant" and "function_call" in message:
            return FunctionCallMessage(**message)
        return Message(**message)

    @property
    def is_assistant_function_call_message(self) -> bool:
        return isinstance(self, FunctionCallMessage)

    @property
    def is_function_message(self) -> bool:
        return self.role == "function"

    @property
    def is_user_message(self) -> bool:
        return self.role == "user"


class FunctionCallData(BaseModel):
    thoughts: str
    name: str
    parameters: str


class FunctionCallMessage(Message):
    function_call: FunctionCallData


def load_messages_from_file(file_path: str) -> list[Message]:
    """load messages from local file"""
    messages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            messages.append({"messages": parse_messages(data["messages"]), "tools": data["tools"]})

    return messages


def parse_messages(messages: list[dict]) -> list[Message]:
    """parse messages"""
    result = []
    for index, message in enumerate(messages):
        role = message["role"]
        if role in ["user", "system", "function"]:
            result.append(Message(**message))
        elif role == "assistant":
            if "function_call" in message:
                if isinstance(message["function_call"], str):
                    message["function_call"] = json.loads(message["function_call"])
                if "arguments" in message["function_call"]:
                    message["function_call"]["parameters"] = message["function_call"].pop("arguments")
                result.append(FunctionCallMessage(**message))
            else:
                result.append(Message(**message))
        else:
            print(message)
            raise ValueError("get unexpected messages")
    return result


def group_round_messages(messages: list[Message]) -> list[list[Message]]:
    """group messages by round"""
    rounds_messages, current_round = [], []
    for message in messages:
        current_round.append(message)
        if message.role == "assistant" and not message.is_function_call_message:
            rounds_messages.append(current_round)
            current_round = []

    if current_round:
        rounds_messages.append(current_round)

    result = []
    for round_messages in rounds_messages:
        assert len(round_messages) >= 2
        user_message, final_message = round_messages[0], round_messages[-1]
        assert user_message.role == "user"
        assert final_message.role == "assistant"
        if len(round_messages) == 2:
            result.append((user_message, [], final_message))
            continue

        result.append((user_message, round_messages[1:-1], final_message))

    return result
