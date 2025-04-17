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

import pytest
from pipelines.nodes.prompt.invocation_layer import PromptModelInvocationLayer
from pipelines.nodes.prompt.prompt_model import PromptModel

from .conftest import create_mock_layer_that_supports


class TestPromptModel(unittest.TestCase):
    def test_constructor_with_default_model(self):
        mock_layer = create_mock_layer_that_supports("gpt-3.5-turbo")
        another_layer = create_mock_layer_that_supports("another-model")

        with patch.object(PromptModelInvocationLayer, "invocation_layer_providers", new=[mock_layer, another_layer]):
            model = PromptModel()
            mock_layer.assert_called_once()
            another_layer.assert_not_called()
            model.model_invocation_layer.model_name_or_path = "gpt-3.5-turbo"

    def test_construtor_with_custom_model(self):
        mock_layer = create_mock_layer_that_supports("some-model")
        another_layer = create_mock_layer_that_supports("another-model")

        with patch.object(PromptModelInvocationLayer, "invocation_layer_providers", new=[mock_layer, another_layer]):
            model = PromptModel("another-model")
            mock_layer.assert_not_called()
            another_layer.assert_called_once()
            model.model_invocation_layer.model_name_or_path = "another-model"

    def test_constructor_with_no_supported_model(self):
        with pytest.raises(ValueError, match="Model some-random-model is not supported"):
            PromptModel("some-random-model")
