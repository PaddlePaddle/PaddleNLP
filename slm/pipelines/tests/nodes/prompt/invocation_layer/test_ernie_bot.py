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
from unittest.mock import MagicMock, patch

from pipelines.nodes.prompt.invocation_layer import ErnieBotInvocationLayer


class TestErnieBot(unittest.TestCase):
    @patch("requests.request")
    def test_invoke(self, mock_request):
        mock_response = MagicMock()
        mock_response.text = '{"result": "Hello, how can I help you?"}'
        mock_request.return_value = mock_response

        invocation_layer = ErnieBotInvocationLayer(api_key="fake_api_key", secret_key="fake_api_key")
        output = invocation_layer.invoke(prompt="dummy_prompt")
        self.assertEqual(output, ["Hello, how can I help you?"])
