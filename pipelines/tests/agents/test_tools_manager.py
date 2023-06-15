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
from unittest import mock

from pipelines import Pipeline
from pipelines.agents import Tool, ToolsManager
from pipelines.schema import Answer, Document


class TestToolsManager(unittest.TestCase):
    def setUp(self):
        tools = [
            Tool(name="ToolA", pipeline_or_node=mock.Mock(), description="Tool A Description"),
            Tool(name="ToolB", pipeline_or_node=mock.Mock(), description="Tool B Description"),
        ]
        self.tools_manager = ToolsManager(tools=tools)

    def test_add_tool(self):
        new_tool = Tool(name="ToolC", pipeline_or_node=mock.Mock(), description="Tool C Description")
        self.tools_manager.add_tool(new_tool)
        assert "ToolC" in self.tools_manager.tools
        assert self.tools_manager.tools["ToolC"] == new_tool

    def test_get_tool_names(self):
        assert self.tools_manager.get_tool_names() == "ToolA, ToolB"

    def test_get_tools(self):
        tools = self.tools_manager.get_tools()
        assert len(tools) == 2
        assert tools[0].name == "ToolA"
        assert tools[1].name == "ToolB"

    def test_get_tool_names_with_descriptions(self):
        expected_output = "ToolA: Tool A Description\n" "ToolB: Tool B Description"
        assert self.tools_manager.get_tool_names_with_descriptions() == expected_output

    def test_extract_tool_name_and_tool_input(self):
        examples = [
            "need to find out what city he was born.\nTool: Search\nTool Input: Where was Jeremy McKinnon born",
            "need to find out what city he was born.\n\nTool: Search\n\nTool Input: Where was Jeremy McKinnon born",
            'need to find out what city he was born. Tool: Search Tool Input: "Where was Jeremy McKinnon born"',
        ]
        for example in examples:
            tool_name, tool_input = self.tools_manager.extract_tool_name_and_tool_input(example)
            assert tool_name == "Search" and tool_input == "Where was Jeremy McKinnon born"

        negative_examples = [
            "need to find out what city he was born.",
            "Tool: Search",
            "Tool Input: Where was Jeremy McKinnon born",
            "need to find out what city he was born. Tool: Search",
            "Tool Input: Where was Jeremy McKinnon born",
        ]
        for example in negative_examples:
            tool_name, tool_input = self.tools_manager.extract_tool_name_and_tool_input(example)
            assert tool_name is None and tool_input is None

    def test_invalid_tool_creation(self):
        with self.assertRaises(ValueError):
            Tool(name="Tool-A", pipeline_or_node=mock.Mock(), description="Tool A Description")

    def test_tool_invocation(self):
        # by default for pipelines as tools we look for results key in the output
        p = Pipeline()
        tool = Tool(name="ToolA", pipeline_or_node=p, description="Tool A Description")
        with unittest.mock.patch("pipelines.pipelines.Pipeline.run", return_value={"results": "mock"}):
            assert tool.run("input") == "mock"

        # now fail if results key is not present
        with unittest.mock.patch("pipelines.pipelines.Pipeline.run", return_value={"no_results": "mock"}):
            with self.assertRaises(ValueError):
                assert tool.run("input")

        # now try tool with a correct output variable
        tool = Tool(name="ToolA", pipeline_or_node=p, description="Tool A Description", output_variable="no_results")
        with unittest.mock.patch("pipelines.pipelines.Pipeline.run", return_value={"no_results": "mock_no_results"}):
            assert tool.run("input") == "mock_no_results"

        # try tool that internally returns an Answer object but we extract the string
        tool = Tool(name="ToolA", pipeline_or_node=p, description="Tool A Description")
        with unittest.mock.patch("pipelines.pipelines.Pipeline.run", return_value=[Answer("mocked_answer")]):
            assert tool.run("input") == "mocked_answer"

        # same but for the document
        with unittest.mock.patch("pipelines.pipelines.Pipeline.run", return_value=[Document("mocked_document")]):
            assert tool.run("input") == "mocked_document"

    def test_extract_tool_name_and_tool_multi_line_input(self):
        # new pattern being supported but with backward compatibility for the old
        text = (
            "We need to find out the following information:\n"
            "1. What city was Jeremy McKinnon born in?\n"
            "2. What's the capital of Germany?\n"
            "Tool: Search\n"
            "Tool Input: Where was Jeremy\n McKinnon born\n and where did he grow up?"
        )

        tool_name, tool_input = self.tools_manager.extract_tool_name_and_tool_input(text)
        assert tool_name == "Search" and tool_input == "Where was Jeremy\n McKinnon born\n and where did he grow up?"

        # tool input is empty
        text2 = (
            "We need to find out the following information:\n"
            "1. What city was Jeremy McKinnon born in?\n"
            "2. What's the capital of Germany?\n"
            "Tool: Search\n"
            "Tool Input:"
        )
        tool_name, tool_input = self.tools_manager.extract_tool_name_and_tool_input(text2)
        assert tool_name == "Search" and tool_input == ""

        # Case where the tool name and tool input are provided with extra whitespaces
        text3 = "   Tool:   Search   \n   Tool Input:   What is the tallest building in the world?   "
        tool_name, tool_input = self.tools_manager.extract_tool_name_and_tool_input(text3)
        assert tool_name.strip() == "Search" and tool_input.strip() == "What is the tallest building in the world?"

        # Case where the tool name is provided but the tool input line is not provided at all
        # Tool input is not optional, so this should return None for both tool name and tool input
        text4 = (
            "We need to find out the following information:\n"
            "1. Who is the current president of the United States?\n"
            "Tool: Search\n"
        )
        tool_name, tool_input = self.tools_manager.extract_tool_name_and_tool_input(text4)
        assert tool_name is None and tool_input is None

        # Case where neither the tool name nor the tool input is provided
        text5 = "We need to find out the following information:\n 1. What is the population of India?"
        tool_name, tool_input = self.tools_manager.extract_tool_name_and_tool_input(text5)
        assert tool_name is None and tool_input is None

        # Case where the tool name and tool input are provided with extra whitespaces and new lines
        text6 = "   Tool:   Search   \n   Tool Input:   \nWhat is the tallest \nbuilding in the world?   "
        tool_name, tool_input = self.tools_manager.extract_tool_name_and_tool_input(text6)
        assert tool_name.strip() == "Search" and tool_input.strip() == "What is the tallest \nbuilding in the world?"

    def test_extract_tool_name_and_empty_tool_input(self):
        examples = [
            "need to find out what city he was born.\nTool: Search\nTool Input:",
            "need to find out what city he was born.\nTool: Search\nTool Input:  ",
        ]
        for example in examples:
            tool_name, tool_input = self.tools_manager.extract_tool_name_and_tool_input(example)
            assert tool_name == "Search" and tool_input == ""
