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

import unittest
from typing import List, Tuple

from .testing_utils import LLMTest

split_turn_messages = None


class TestSplitTurnMessages(LLMTest, unittest.TestCase):
    def setUp(self):
        self.fc = dict(name="tool", thoughts="I should use tool-a", parameters="...")
        super().setUp()
        global split_turn_messages
        from function_call.utils import split_turn_messages

    def create_messages(self, roles: List[str], contents: List[str]):
        from function_call.schema import Message

        messages = []
        for role, content in zip(roles, contents):
            if role in ["user", "assistant", "function"]:
                messages.append(Message.from_dict(role=role, content=content))
            else:
                messages.append(Message.from_dict(role="assistant", function_call=content))

        return messages

    def test_split_with_single_turn(self):
        messages = self.create_messages(
            ["user", "function_call", "function", "assistant"], ["Hello", self.fc, "B1", "Assistant Response"]
        )
        expected = [(messages[0], [messages[1], messages[2]], messages[3])]
        self.assertEqual(split_turn_messages(messages), expected)

    def test_split_with_multiple_turns(self):
        messages = self.create_messages(
            ["user", "function_call", "function", "assistant", "user", "function_call", "function", "assistant"],
            ["Hello", self.fc, "B1", "Assistant Response 1", "Hi", self.fc, "B2", "Assistant Response 2"],
        )
        expected = [
            (messages[0], [messages[1], messages[2]], messages[3]),
            (messages[4], [messages[5], messages[6]], messages[7]),
        ]
        self.assertEqual(split_turn_messages(messages), expected)

    def test_split_with_multiple_ab_pairs(self):
        messages = self.create_messages(
            ["user", "function_call", "function", "function_call", "function", "assistant"],
            ["Hello", self.fc, "B1", self.fc, "B2", "Assistant Response"],
        )
        expected = [(messages[0], [messages[1], messages[2], messages[3], messages[4]], messages[5])]
        self.assertEqual(split_turn_messages(messages), expected)

    def test_split_with_missing_assistant(self):
        messages = self.create_messages(["user", "function_call", "function"], ["Hello", self.fc, "B1"])
        expected = [(messages[0], [messages[1], messages[2]], None)]
        self.assertEqual(split_turn_messages(messages), expected)

    def test_split_with_missing_ab(self):
        messages = self.create_messages(["user", "assistant"], ["Hello", "Assistant Response"])
        expected = [(messages[0], [], messages[1])]
        self.assertEqual(split_turn_messages(messages), expected)

    def test_split_with_empty_input(self):
        messages = []
        expected = []
        self.assertEqual(split_turn_messages(messages), expected)
