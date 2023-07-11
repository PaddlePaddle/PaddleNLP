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

from pipelines.agents import AgentStep
from pipelines.schema import Answer


class TestAgentSteps(unittest.TestCase):
    def setUp(self):
        self.agent_step = AgentStep(
            current_step=1, max_steps=10, final_answer_pattern=None, prompt_node_response="Hello", transcript="Hello"
        )

    def test_create_next_step(self):
        # Test normal case
        next_step = self.agent_step.create_next_step(["Hello again"])
        assert next_step.current_step == 2
        assert next_step.prompt_node_response == "Hello again"
        assert next_step.transcript == "Hello"

        # Test with invalid prompt_node_response
        with self.assertRaises(Exception):
            self.agent_step.create_next_step({})

        # Test with empty prompt_node_response
        with self.assertRaises(Exception):
            self.agent_step.create_next_step([])

    def test_final_answer(self):
        # Test normal case
        result = self.agent_step.final_answer("query")
        assert result["query"] == "query"
        assert isinstance(result["answers"][0], Answer)
        assert result["answers"][0].answer == "Hello"
        assert result["answers"][0].type == "generative"
        assert result["transcript"] == "Hello"

        # Test with max_steps reached
        self.agent_step.current_step = 11
        result = self.agent_step.final_answer("query")
        assert result["answers"][0].answer == ""

    def test_is_last(self):
        # Test is last, and it is last because of valid prompt_node_response and default final_answer_pattern
        agent_step = AgentStep(current_step=1, max_steps=10, prompt_node_response="Hello", transcript="Hello")
        assert agent_step.is_last()

        # Test not last
        agent_step.current_step = 1
        agent_step.prompt_node_response = "final answer not satisfying pattern"
        agent_step.final_answer_pattern = r"Final Answer\s*:\s*(.*)"
        assert not agent_step.is_last()

        # Test border cases for max_steps
        agent_step.current_step = 9
        assert not agent_step.is_last()
        agent_step.current_step = 10
        assert not agent_step.is_last()

        # Test when last due to max_steps
        agent_step.current_step = 11
        assert agent_step.is_last()

    def test_completed(self):
        # Test without observation
        self.agent_step.completed(None)
        assert self.agent_step.transcript == "HelloHello"

        # Test with observation, adds Hello from prompt_node_response
        self.agent_step.completed("observation")
        assert self.agent_step.transcript == "HelloHelloHello\nObservation: observation\nThought:"

    def test_repr(self):
        assert repr(self.agent_step) == (
            "AgentStep(current_step=1, max_steps=10, "
            "prompt_node_response=Hello, final_answer_pattern=^([\\s\\S]+)$, "
            "transcript=Hello)"
        )

    def test_parse_final_answer(self):
        # Test when pattern matches
        assert self.agent_step.parse_final_answer() == "Hello"

        # Test when pattern does not match
        self.agent_step.final_answer_pattern = "goodbye"
        assert self.agent_step.parse_final_answer() is None

    def test_format_react_answer(self):
        step = AgentStep(
            final_answer_pattern=r"Final Answer\s*:\s*(.*)",
            prompt_node_response="have the final answer to the question.\nFinal Answer: Florida",
        )
        formatted_answer = step.final_answer(query="query")
        assert formatted_answer["query"] == "query"
        assert formatted_answer["answers"] == [Answer(answer="Florida", type="generative")]
