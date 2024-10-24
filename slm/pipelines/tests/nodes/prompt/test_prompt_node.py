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
from unittest.mock import Mock, patch

import pytest
from pipelines.nodes.prompt import PromptNode, PromptTemplate


class TestPromptNode(unittest.TestCase):
    def test_add_and_remove_template(self):
        with patch("pipelines.nodes.prompt.prompt_node.PromptModel"):
            node = PromptNode()
        total_count = 15
        # Verifies default
        assert len(node.get_prompt_template_names()) == total_count

        # Add a fake template
        fake_template = PromptTemplate(name="fake-template", prompt_text="Fake prompt")
        node.add_prompt_template(fake_template)
        assert len(node.get_prompt_template_names()) == total_count + 1
        assert "fake-template" in node.get_prompt_template_names()

        # Verify that adding the same template throws an expection
        with pytest.raises(ValueError) as e:
            node.add_prompt_template(fake_template)
            assert e.match(
                "Prompt template fake-template already exists. Select a different name for this prompt template."
            )

        # Verify template is correctly removed
        assert node.remove_prompt_template("fake-template")
        assert len(node.get_prompt_template_names()) == total_count
        assert "fake-template" not in node.get_prompt_template_names()

        # Verify that removing the same template throws an expection
        with pytest.raises(ValueError) as e:
            node.remove_prompt_template("fake-template")
            assert e.match("Prompt template fake-template does not exist")

    @patch.object(PromptNode, "prompt")
    @patch("pipelines.nodes.prompt.prompt_node.PromptModel")
    def test_prompt_call_with_no_kwargs(self, mock_model, mocked_prompt):
        node = PromptNode()
        node()
        mocked_prompt.assert_called_once_with(node.default_prompt_template)

    @patch.object(PromptNode, "prompt")
    @patch("pipelines.nodes.prompt.prompt_node.PromptModel")
    def test_prompt_call_with_custom_kwargs(self, mock_model, mocked_prompt):
        node = PromptNode()
        node(some_kwarg="some_value")
        mocked_prompt.assert_called_once_with(node.default_prompt_template, some_kwarg="some_value")

    @patch.object(PromptNode, "prompt")
    @patch("pipelines.nodes.prompt.prompt_node.PromptModel")
    def test_prompt_call_with_custom_template(self, mock_model, mocked_prompt):
        node = PromptNode()
        mock_template = Mock()
        node(prompt_template=mock_template)
        mocked_prompt.assert_called_once_with(mock_template)

    @patch.object(PromptNode, "prompt")
    @patch("pipelines.nodes.prompt.prompt_node.PromptModel")
    def test_prompt_call_with_custom_kwargs_and_template(self, mock_model, mocked_prompt):
        node = PromptNode()
        mock_template = Mock()
        node(prompt_template=mock_template, some_kwarg="some_value")
        mocked_prompt.assert_called_once_with(mock_template, some_kwarg="some_value")

    @patch("pipelines.nodes.prompt.prompt_node.PromptModel")
    def test_get_prompt_template_without_default_template(self, mock_model):
        node = PromptNode()
        assert node.get_prompt_template() is None

        template = node.get_prompt_template("question-answering")
        assert template.name == "question-answering"

        template = node.get_prompt_template(PromptTemplate(name="fake-template", prompt_text=""))
        assert template.name == "fake-template"

        with pytest.raises(ValueError) as e:
            node.get_prompt_template("some-unsupported-template")
            assert e.match("some-unsupported-template not supported, select one of:")

        fake_yaml_prompt = "name: fake-yaml-template\nprompt_text: fake prompt text"
        template = node.get_prompt_template(fake_yaml_prompt)
        assert template.name == "fake-yaml-template"

        fake_yaml_prompt = "- prompt_text: fake prompt text"
        template = node.get_prompt_template(fake_yaml_prompt)
        assert template.name == "custom-at-query-time"

        template = node.get_prompt_template("some prompt")
        assert template.name == "custom-at-query-time"

    @patch("pipelines.nodes.prompt.prompt_node.PromptModel")
    def test_get_prompt_template_with_default_template(self, mock_model):
        node = PromptNode()
        node.set_default_prompt_template("question-answering")

        template = node.get_prompt_template()
        assert template.name == "question-answering"

        template = node.get_prompt_template("sentiment-analysis")
        assert template.name == "sentiment-analysis"

        template = node.get_prompt_template(PromptTemplate(name="fake-template", prompt_text=""))
        assert template.name == "fake-template"

        with pytest.raises(ValueError) as e:
            node.get_prompt_template("some-unsupported-template")
            assert e.match("some-unsupported-template not supported, select one of:")

        fake_yaml_prompt = "name: fake-yaml-template\nprompt_text: fake prompt text"
        template = node.get_prompt_template(fake_yaml_prompt)
        assert template.name == "fake-yaml-template"

        fake_yaml_prompt = "- prompt_text: fake prompt text"
        template = node.get_prompt_template(fake_yaml_prompt)
        assert template.name == "custom-at-query-time"

        template = node.get_prompt_template("some prompt")
        assert template.name == "custom-at-query-time"

    @patch("pipelines.nodes.prompt.prompt_node.PromptModel")
    def test_prompt_node_streaming_handler_on_call(self, mock_model):
        """
        Verifies model is created using expected stream handler when calling PromptNode.
        """
        mock_handler = Mock()
        node = PromptNode()
        node.prompt_model = mock_model
        node("Irrelevant prompt", stream=True, stream_handler=mock_handler)
        # Verify model has been constructed with expected model_kwargs
        mock_model.invoke.assert_called_once()
        assert mock_model.invoke.call_args_list[0].kwargs["stream_handler"] == mock_handler

    @patch("pipelines.nodes.prompt.prompt_node.PromptModel")
    def test_prompt_node_streaming_handler_on_constructor(self, mock_model):
        """
        Verifies model is created using expected stream handler when constructing PromptNode.
        """
        model_kwargs = {"stream_handler": Mock()}
        PromptNode(model_kwargs=model_kwargs)
        # Verify model has been constructed with expected model_kwargs
        mock_model.assert_called_once()
        assert mock_model.call_args_list[0].kwargs["model_kwargs"] == model_kwargs
