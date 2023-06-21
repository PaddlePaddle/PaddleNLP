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
from unittest.mock import patch

from pipelines.nodes.prompt.invocation_layer import OpenAIInvocationLayer


class TestOpenAI(unittest.TestCase):
    @patch("pipelines.nodes.prompt.invocation_layer.open_ai.openai_request")
    def test_default_api_base(self, mock_request):
        invocation_layer = OpenAIInvocationLayer(api_key="fake_api_key")
        assert invocation_layer.api_base == "https://api.openai.com/v1"
        assert invocation_layer.url == "https://api.openai.com/v1/completions"

        invocation_layer.invoke(prompt="dummy_prompt")
        assert mock_request.call_args.kwargs["url"] == "https://api.openai.com/v1/completions"

    @patch("pipelines.nodes.prompt.invocation_layer.open_ai.openai_request")
    def test_custom_api_base(self, mock_request):
        invocation_layer = OpenAIInvocationLayer(api_key="fake_api_key", api_base="https://fake_api_base.com")
        assert invocation_layer.api_base == "https://fake_api_base.com"
        assert invocation_layer.url == "https://fake_api_base.com/completions"

        invocation_layer.invoke(prompt="dummy_prompt")
        assert mock_request.call_args.kwargs["url"] == "https://fake_api_base.com/completions"
