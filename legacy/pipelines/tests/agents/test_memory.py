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

import unittest
from typing import Any, Dict

from pipelines.agents.memory import ConversationMemory, NoMemory


class TestMemory(unittest.TestCase):
    def test_no_memory(self):
        no_mem = NoMemory()
        assert no_mem.load() == ""
        no_mem.save({"key": "value"})
        no_mem.clear()

    def test_conversation_memory(self):
        conv_mem = ConversationMemory()
        assert conv_mem.load() == ""
        data: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
        conv_mem.save(data)
        assert conv_mem.load() == "Human: Hello\nAI: Hi there\n"

        data: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
        conv_mem.save(data)
        assert conv_mem.load() == "Human: Hello\nAI: Hi there\nHuman: How are you?\nAI: I'm doing well, thanks.\n"
        assert conv_mem.load(window_size=1) == "Human: How are you?\nAI: I'm doing well, thanks.\n"

        conv_mem.clear()
        assert conv_mem.load() == ""

    def test_conversation_memory_window_size(self):
        conv_mem = ConversationMemory()
        assert conv_mem.load() == ""
        data: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
        conv_mem.save(data)
        data: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
        conv_mem.save(data)
        assert conv_mem.load() == "Human: Hello\nAI: Hi there\nHuman: How are you?\nAI: I'm doing well, thanks.\n"
        assert conv_mem.load(window_size=1) == "Human: How are you?\nAI: I'm doing well, thanks.\n"

        # clear the memory
        conv_mem.clear()
        assert conv_mem.load() == ""
        assert conv_mem.load(window_size=1) == ""
