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

import re
import unittest
from unittest import mock

from pipelines import Answer, BaseComponent, Document
from pipelines.agents import Agent, AgentStep
from pipelines.agents.base import Tool
from pipelines.nodes.prompt import PromptNode
from pipelines.pipelines import BaseStandardPipeline, SemanticSearchPipeline

from tests.conftest import MockPromptNode, MockRetriever


class TestAgents(unittest.TestCase):
    def test_add_and_overwrite_tool(self):
        # Add a Node as a Tool to an Agent
        agent = Agent(prompt_node=MockPromptNode())
        retriever = MockRetriever()
        agent.add_tool(
            Tool(
                name="Retriever",
                pipeline_or_node=retriever,
                description="useful for when you need to " "retrieve documents from your index",
            )
        )
        assert len(agent.tm.tools) == 1
        assert agent.has_tool(tool_name="Retriever")
        assert isinstance(agent.tm.tools["Retriever"].pipeline_or_node, BaseComponent)

        agent.add_tool(
            Tool(
                name="Retriever",
                pipeline_or_node=retriever,
                description="useful for when you need to retrieve documents from your index",
            )
        )

        # Add a Pipeline as a Tool to an Agent and overwrite the previously added Tool
        retriever_pipeline = SemanticSearchPipeline(MockRetriever())
        agent.add_tool(
            Tool(
                name="Retriever",
                pipeline_or_node=retriever_pipeline,
                description="useful for when you need to retrieve documents from your index",
            )
        )
        assert len(agent.tm.tools) == 1
        assert agent.has_tool(tool_name="Retriever")
        assert isinstance(agent.tm.tools["Retriever"].pipeline_or_node, BaseStandardPipeline)

    def test_run_tool(self):
        agent = Agent(prompt_node=MockPromptNode())
        retriever = MockRetriever()
        agent.add_tool(
            Tool(
                name="Retriever",
                pipeline_or_node=retriever,
                description="useful for when you need to retrieve documents from your index",
                output_variable="documents",
            )
        )
        pn_response = (
            "need to find out what city he was born.\nTool: Retriever\nTool Input: Where was Jeremy McKinnon born"
        )

        step = AgentStep(prompt_node_response=pn_response)
        result = agent.tm.run_tool(step.prompt_node_response)
        assert result == "[]"  # empty list of documents

    def test_extract_final_answer(self):
        match_examples = [
            "have the final answer to the question.\nFinal Answer: Florida",
            "Final Answer: 42 is the answer",
            "Final Answer:  1234",
            "Final Answer:  Answer",
            "Final Answer:  This list: one and two and three",
            "Final Answer:42",
            "Final Answer:   ",
            "Final Answer:    The answer is 99    ",
        ]
        expected_answers = [
            "Florida",
            "42 is the answer",
            "1234",
            "Answer",
            "This list: one and two and three",
            "42",
            "",
            "The answer is 99",
        ]

        for example, expected_answer in zip(match_examples, expected_answers):
            agent_step = AgentStep(prompt_node_response=example, final_answer_pattern=r"Final Answer\s*:\s*(.*)")
            final_answer = agent_step.final_answer(query="irrelevant")
            assert final_answer["answers"][0].answer == expected_answer

    def test_final_answer_regex(self):
        match_examples = [
            "Final Answer: 42 is the answer",
            "Final Answer:  1234",
            "Final Answer:  Answer",
            "Final Answer:  This list: one and two and three",
            "Final Answer:42",
            "Final Answer:   ",
            "Final Answer:    The answer is 99    ",
        ]

        non_match_examples = ["Final answer: 42 is the answer", "Final Answer", "The final answer is: 100"]
        final_answer_pattern = r"Final Answer\s*:\s*(.*)"
        for example in match_examples:
            match = re.match(final_answer_pattern, example)
            assert match is not None

        for example in non_match_examples:
            match = re.match(final_answer_pattern, example)
            assert match is None

    def test_update_hash(
        self,
    ):
        agent = Agent(prompt_node=MockPromptNode(), prompt_template=mock.Mock())
        assert agent.hash == "d41d8cd98f00b204e9800998ecf8427e"
        agent.add_tool(
            Tool(
                name="Search",
                pipeline_or_node=mock.Mock(),
                description="useful for when you need to answer "
                "questions about where people live. You "
                "should ask targeted questions",
                output_variable="answers",
            )
        )
        assert agent.hash == "d41d8cd98f00b204e9800998ecf8427e"
        agent.add_tool(
            Tool(
                name="Count",
                pipeline_or_node=mock.Mock(),
                description="useful for when you need to count how many characters are in a word. Ask only with a single word.",
            )
        )
        assert agent.hash == "d41d8cd98f00b204e9800998ecf8427e"
        agent.update_hash()
        assert agent.hash == "5ac8eca2f92c9545adcce3682b80d4c5"

    def test_tool_fails_processing_dict_result(self):
        tool = Tool(name="name", pipeline_or_node=MockPromptNode(), description="description")
        with self.assertRaises(ValueError):
            tool._process_result({"answer": "answer"})

    def test_tool_processes_answer_result_and_document_result(self):
        tool = Tool(name="name", pipeline_or_node=MockPromptNode(), description="description")
        assert tool._process_result(Answer(answer="answer")) == "answer"
        assert tool._process_result(Document(content="content")) == "content"

    def test_invalid_agent_template(self):
        pn = PromptNode("gpt-3.5-turbo", api_key="12345678")
        with self.assertRaises(ValueError):
            Agent(prompt_node=pn, prompt_template="some_non_existing_template")

        # # if prompt_template is None, then we'll use zero-shot-react
        # a = Agent(prompt_node=pn, prompt_template=None)
        # assert isinstance(a.prompt_template, PromptTemplate)
        # assert a.prompt_template.name == "zero-shot-react"
