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

import collections
from typing import Any, Dict, List, Optional, OrderedDict

from pipelines.agents.memory import Memory


class ConversationMemory(Memory):
    """
    A memory class that stores conversation history.
    """

    def __init__(self, input_key: str = "input", output_key: str = "output"):
        """
        Initialize ConversationMemory with input and output keys.

        :param input_key: The key to use for storing user input.
        :param output_key: The key to use for storing model output.
        """
        self.list: List[OrderedDict] = []
        self.input_key = input_key
        self.output_key = output_key

    def load(self, keys: Optional[List[str]] = None, **kwargs) -> str:
        """
        Load conversation history as a formatted string.

        :param keys: Optional list of keys (ignored in this implementation).
        :param kwargs: Optional keyword arguments
            - window_size: integer specifying the number of most recent conversation snippets to load.
        :return: A formatted string containing the conversation history.
        """
        chat_transcript = ""
        window_size = kwargs.get("window_size", None)

        if window_size is not None:
            chat_list = self.list[-window_size:]  # pylint: disable=invalid-unary-operand-type
        else:
            chat_list = self.list

        for chat_snippet in chat_list:
            chat_transcript += f"Human: {chat_snippet['Human']}\n"
            chat_transcript += f"AI: {chat_snippet['AI']}\n"
        return chat_transcript

    def save(self, data: Dict[str, Any]) -> None:
        """
        Save a conversation snippet to memory.

        :param data: A dictionary containing the conversation snippet to save.
        """
        chat_snippet = collections.OrderedDict()
        chat_snippet["Human"] = data[self.input_key]
        chat_snippet["AI"] = data[self.output_key]
        self.list.append(chat_snippet)

    def clear(self) -> None:
        """
        Clear the conversation history.
        """
        self.list = []
