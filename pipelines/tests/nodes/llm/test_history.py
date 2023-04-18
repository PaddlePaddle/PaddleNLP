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

from pipelines.nodes.llm import TruncatedConversationHistory


class TestTruncatedConversationHistory(unittest.TestCase):
    def test_truncated_conversation_history(self):
        max_length = 10
        component = TruncatedConversationHistory(max_length)
        query = "test_query"
        history = [
            {"content": "This is a very long message."},
            {"content": "Short message."},
            {"content": "This one is too long as well, let's see what happens."},
        ]

        expected_history = [
            {"content": "This is a "},
            {"content": "Short mess"},
            {"content": "This one i"},
        ]

        truncated_history, output_key = component.run(query=query, history=history)

        self.assertEqual(truncated_history["query"], query)
        self.assertEqual(truncated_history["history"], expected_history)
        self.assertEqual(output_key, "output_1")

    def test_no_history(self):
        max_length = 10
        component = TruncatedConversationHistory(max_length)
        query = "test_query"

        truncated_history, output_key = component.run(query=query)

        self.assertEqual(truncated_history["query"], query)
        self.assertNotIn("history", truncated_history)
        self.assertEqual(output_key, "output_1")
