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

import os
import unittest
from unittest.mock import MagicMock, patch

from pipelines.nodes.llm import ErnieBot


class TestErnieBot(unittest.TestCase):
    def setUp(self):
        self.api_key = "your api_key"
        self.secret_key = "your secret_key"
        self.valid_history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]

    @patch("requests.request")
    def test_init_with_access_token(self, mock_request):
        mock_response = MagicMock()
        mock_response.text = '{"access_key": "1234567890"}'
        mock_request.return_value = mock_response
        ernie_bot = ErnieBot(api_key=self.api_key, secret_key=self.secret_key)
        self.assertIsNotNone(ernie_bot)

    def test_init_missing_access_token(self):
        os.environ.pop("ERNIE_BOT_API_KEY", None)
        os.environ.pop("ERNIE_BOT_SECRET_KEY", None)
        with self.assertRaises(Exception):
            ErnieBot()

    @patch("requests.request")
    def test_run_without_history(self, mock_request):
        mock_response = MagicMock()
        mock_response.text = '{"result": "Hello, how can I help you?"}'
        mock_request.return_value = mock_response

        ernie_bot = ErnieBot(api_key=self.api_key, secret_key=self.secret_key)
        response, _ = ernie_bot.run("Hi")

        self.assertEqual(
            response["history"],
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello, how can I help you?"}],
        )

    @patch("requests.request")
    def test_run_with_valid_history(self, mock_request):
        mock_response = MagicMock()
        mock_response.text = '{"result": "I can help you with that."}'
        mock_request.return_value = mock_response

        ernie_bot = ErnieBot(api_key=self.api_key, secret_key=self.secret_key)
        response, _ = ernie_bot.run("What can you do?", history=self.valid_history)
        self.assertEqual(
            response["history"],
            self.valid_history
            + [
                {"role": "user", "content": "What can you do?"},
                {"role": "assistant", "content": "I can help you with that."},
            ],
        )

    @patch("requests.request")
    def test_run_with_invalid_history_role(self, mock_request):
        mock_response = MagicMock()
        mock_response.text = '{"result": "I can help you with that."}'
        mock_request.return_value = mock_response
        invalid_history = [{"role": "invalid", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]

        ernie_bot = ErnieBot(api_key=self.api_key, secret_key=self.secret_key)
        with self.assertRaises(ValueError):
            ernie_bot.run("What can you do?", history=invalid_history)

    @patch("requests.request")
    def test_run_with_odd_history_length(self, mock_request):
        mock_response = MagicMock()
        mock_response.text = '{"result": "I can help you with that."}'
        mock_request.return_value = mock_response

        odd_history = [{"role": "user", "content": "Hi"}]

        ernie_bot = ErnieBot(api_key=self.api_key, secret_key=self.secret_key)
        with self.assertRaises(ValueError):
            ernie_bot.run("What can you do?", history=odd_history)
