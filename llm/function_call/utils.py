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

from typing import List, Tuple, Type

from .schema import FunctionCallMessage, Message

TurnMessage = Tuple[Message, List[Message], Message]


def split_turn_messages(messages: List[Message]) -> List[TurnMessage]:
    result = []
    user_message = None
    tool_messages = []
    assistant_message = None

    for message in messages:
        if message.role == "user":
            if user_message or tool_messages or assistant_message:
                # 如果已经有累积的数据，则先保存，再清空准备新一轮的收集
                result.append((user_message, tool_messages, assistant_message))
                user_message = None
                tool_messages = []
                assistant_message = None
            user_message = message
        elif message.is_function_call_message or message.is_assistant_function_call_message:
            tool_messages.append(message)
        elif message.role == "assistant":
            assistant_message = message

    # 添加最后一组数据（如果有的话）
    if user_message or tool_messages or assistant_message:
        result.append((user_message, tool_messages, assistant_message))

    return result
