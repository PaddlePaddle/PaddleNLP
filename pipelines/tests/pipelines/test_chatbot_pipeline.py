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
import unittest
from unittest.mock import MagicMock, patch

from pipelines.nodes.llm import ChatGLMBot, ErnieBot, TruncatedConversationHistory
from pipelines.pipelines import Pipeline


class TestChatPipeline(unittest.TestCase):
    @patch("requests.request")
    def setUp(self, mock_request):
        self.eb = ErnieBot(api_key="api_key", secret_key="secret_key")

    def request_side_effect(*args, **kwargs):
        data = json.loads(kwargs["data"])
        num_messages = len(data["messages"])
        response_text = json.dumps({"result": f"{num_messages} messages received"})
        mock_response = MagicMock()
        mock_response.text = response_text
        return mock_response

    @patch("requests.request")
    def test_run_with_history(self, mock_request):
        mock_request.side_effect = self.request_side_effect
        pipeline = Pipeline()
        pipeline.add_node(component=self.eb, name="ErnieBot", inputs=["Query"])

        response_round_1 = pipeline.run("hello")
        self.assertEqual(response_round_1["result"], "1 messages received")
        self.assertEqual(
            response_round_1["history"],
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "1 messages received"}],
        )

        response_round_2 = pipeline.run("hello again", params={"history": response_round_1["history"]})
        self.assertEqual(response_round_2["result"], "3 messages received")
        self.assertEqual(
            response_round_2["history"],
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "1 messages received"},
                {"role": "user", "content": "hello again"},
                {"role": "assistant", "content": "3 messages received"},
            ],
        )

    @patch("requests.request")
    def test_run_with_truncated_history(self, mock_request):
        mock_request.side_effect = self.request_side_effect
        pipeline = Pipeline()
        pipeline.add_node(
            component=TruncatedConversationHistory(max_length=1), name="TruncateHistory", inputs=["Query"]
        )
        pipeline.add_node(component=self.eb, name="ErnieBot", inputs=["TruncateHistory"])

        response_round_1 = pipeline.run("hello")
        self.assertEqual(response_round_1["result"], "1 messages received")
        self.assertEqual(
            response_round_1["history"],
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "1 messages received"}],
        )

        response_round_2 = pipeline.run("hello again", history=response_round_1["history"])
        self.assertEqual(response_round_2["result"], "3 messages received")
        self.assertEqual(
            response_round_2["history"],
            [
                {"role": "user", "content": "h"},
                {"role": "assistant", "content": "1"},
                {"role": "user", "content": "hello again"},
                {"role": "assistant", "content": "3 messages received"},
            ],
        )


class TestChatGLMPipeline(unittest.TestCase):
    def setUp(self):
        self.chat = ChatGLMBot(
            model_name_or_path="__internal_testing__/tiny-random-chatglm", dtype="float32", tgt_length=8
        )

    def test_run_without_history(self):
        pipeline = Pipeline()
        pipeline.add_node(component=self.chat, name="ChatBot", inputs=["Query"])
        response_round_1 = pipeline.run("hello")
        self.assertEqual(response_round_1["result"], ["strained睡到睡到睡到睡到睡到睡到睡到"])
