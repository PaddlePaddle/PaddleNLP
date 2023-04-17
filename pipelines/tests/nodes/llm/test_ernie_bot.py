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
import os
import unittest
from unittest.mock import patch

from pipelines.nodes.llm import ErnieBot


class TestErnieBot(unittest.TestCase):
    def test_missing_access_token(self):
        with self.assertRaises(ValueError):
            ErnieBot()

    @patch.dict(os.environ, {"ernie_bot_access_token": "test_token"})
    def test_access_token_from_os_environ(self):
        os.environ["ernie_bot_access_token"] = "test_tokens123"
        ErnieBot()

    @patch("requests.request")
    def test_run(self, mock_request):
        # Mock the API response
        mock_response = {
            "id": "as-bcmt5ct4iy",
            "object": "chat.completion",
            "created": 1680167072,
            "result": "您好，我是百度研发的知识增强大语言模型，中文名是文心一言，英文名是ERNIE Bot。我能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。",
            "need_clear_history": False,
            "usage": {"prompt_tokens": 7, "completion_tokens": 67, "total_tokens": 74},
        }

        # Configure the mock to return the response
        mock_request.return_value.text = json.dumps(mock_response)

        ernie_bot = ErnieBot(ernie_bot_access_token="test_token")
        sample_query = "介绍一下你自己"
        result, output_key = ernie_bot.run(sample_query)

        # Assert that the function call returns the expected output
        self.assertEqual(result, mock_response)
        self.assertEqual(output_key, "eb_output")

        # Assert that the mock was called with the correct parameters
        mock_request.assert_called_once_with(
            "POST",
            ernie_bot.url,
            headers=ernie_bot.headers,
            data=json.dumps({"messages": [{"role": "user", "content": sample_query}]}),
        )
