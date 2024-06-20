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

from pipelines.nodes.prompt import PromptTemplate
from pipelines.nodes.prompt.prompt_node import PromptNode
from pipelines.nodes.prompt.shapers import AnswerParser
from pipelines.pipelines.base import Pipeline


class TestPromptTemplate(unittest.TestCase):
    def test_prompt_templates(self):
        p = PromptTemplate("t1", "Here is some fake template with variable {foo}")
        assert set(p.prompt_params) == {"foo"}

        p = PromptTemplate("t3", "Here is some fake template with variable {foo} and {bar}")
        assert set(p.prompt_params) == {"foo", "bar"}

        p = PromptTemplate("t4", "Here is some fake template with variable {foo1} and {bar2}")
        assert set(p.prompt_params) == {"foo1", "bar2"}

        p = PromptTemplate("t4", "Here is some fake template with variable {foo_1} and {bar_2}")
        assert set(p.prompt_params) == {"foo_1", "bar_2"}

        p = PromptTemplate("t4", "Here is some fake template with variable {Foo_1} and {Bar_2}")
        assert set(p.prompt_params) == {"Foo_1", "Bar_2"}

        p = PromptTemplate("t4", "'Here is some fake template with variable {baz}'")
        assert set(p.prompt_params) == {"baz"}
        # strip single quotes, happens in YAML as we need to use single quotes for the template string
        assert p.prompt_text == "Here is some fake template with variable {baz}"

        p = PromptTemplate("t4", '"Here is some fake template with variable {baz}"')
        assert set(p.prompt_params) == {"baz"}
        # strip double quotes, happens in YAML as we need to use single quotes for the template string
        assert p.prompt_text == "Here is some fake template with variable {baz}"

    def test_missing_prompt_template_params(self):
        template = PromptTemplate("missing_params", "Here is some fake template with variable {foo} and {bar}")

        # both params provided - ok
        template.prepare(foo="foo", bar="bar")

        # missing one param
        with self.assertRaises(ValueError):
            template.prepare(foo="foo")

        # missing both params
        with self.assertRaises(ValueError):
            template.prepare(lets="go")

        # more than both params provided - also ok
        template.prepare(foo="foo", bar="bar", lets="go")

    def test_prompt_template_repr(self):
        p = PromptTemplate("t", "Here is variable {baz}")
        desired_repr = "PromptTemplate(name=t, prompt_text=Here is variable {baz}, prompt_params=['baz'])"
        assert repr(p) == desired_repr
        assert str(p) == desired_repr

    @patch("pipelines.nodes.prompt.prompt_node.PromptModel")
    def test_prompt_template_deserialization(self, mock_prompt_model):
        custom_prompt_template = PromptTemplate(
            name="custom-question-answering",
            prompt_text="Given the context please answer the question. Context: {context}; Question: {query}; Answer:",
            output_parser=AnswerParser(),
        )

        prompt_node = PromptNode(default_prompt_template=custom_prompt_template)

        pipe = Pipeline()
        pipe.add_node(component=prompt_node, name="Generator", inputs=["Query"])
        # TODO(wugaosheng): compatible to config
        # config = pipe.get_config()
        # loaded_pipe = Pipeline.load_from_config(config)

        loaded_generator = pipe.get_node("Generator")
        assert isinstance(loaded_generator, PromptNode)
        assert isinstance(loaded_generator.default_prompt_template, PromptTemplate)
        assert loaded_generator.default_prompt_template.name == "custom-question-answering"
        assert (
            loaded_generator.default_prompt_template.prompt_text
            == "Given the context please answer the question. Context: {context}; Question: {query}; Answer:"
        )
        assert isinstance(loaded_generator.default_prompt_template.output_parser, AnswerParser)
