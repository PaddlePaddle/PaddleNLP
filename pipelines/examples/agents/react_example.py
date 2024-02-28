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

import argparse

from pipelines.agents import Agent, Tool
from pipelines.agents.base import ToolsManager
from pipelines.nodes import PromptNode, WebRetriever
from pipelines.nodes.prompt.prompt_template import PromptTemplate
from pipelines.pipelines import WebQAPipeline

few_shot_prompt = """
You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions correctly, you have access to the following tools:

Search: useful for when you need to Google questions. You should ask targeted questions, for example, Who is Anthony Dirrell's brother?

To answer questions, you'll need to go through multiple steps involving step-by-step thinking and selecting appropriate tools and their inputs; tools will respond with observations. When you are ready for a final answer, respond with the `Final Answer:`
Examples:
##
Question: Anthony Dirrell is the brother of which super middleweight title holder?
Thought: Let's think step by step. To answer this question, we first need to know who Anthony Dirrell is.
Tool: Search
Tool Input: Who is Anthony Dirrell?
Observation: Boxer
Thought: We've learned Anthony Dirrell is a Boxer. Now, we need to find out who his brother is.
Tool: Search
Tool Input: Who is Anthony Dirrell brother?
Observation: Andre Dirrell
Thought: We've learned Andre Dirrell is Anthony Dirrell's brother. Now, we need to find out what title Andre Dirrell holds.
Tool: Search
Tool Input: What is the Andre Dirrell title?
Observation: super middleweight
Thought: We've learned Andre Dirrell title is super middleweight. Now, we can answer the question.
Final Answer: Andre Dirrell
##
Question: What year was the party of the winner of the 1971 San Francisco mayoral election founded?
Thought: Let's think step by step. To answer this question, we first need to know who won the 1971 San Francisco mayoral election.
Tool: Search
Tool Input: Who won the 1971 San Francisco mayoral election?
Observation: Joseph Alioto
Thought: We've learned Joseph Alioto won the 1971 San Francisco mayoral election. Now, we need to find out what party he belongs to.
Tool: Search
Tool Input: What party does Joseph Alioto belong to?
Observation: Democratic Party
Thought: We've learned Democratic Party is the party of Joseph Alioto. Now, we need to find out when the Democratic Party was founded.
Tool: Search
Tool Input: When was the Democratic Party founded?
Observation: 1828
Thought: We've learned the Democratic Party was founded in 1828. Now, we can answer the question.
Final Answer: 1828
##
Question: Right Back At It Again contains lyrics co-written by the singer born in what city?
Thought: Let's think step by step. To answer this question, we first need to know what song the question is referring to.
Tool: Search
Tool Input: What is the song Right Back At It Again?
Observation: "Right Back at It Again" is the song by A Day to Remember
Thought: We've learned Right Back At It Again is a song by A Day to Remember. Now, we need to find out who co-wrote the song.
Tool: Search
Tool Input: Who co-wrote the song Right Back At It Again?
Observation: Jeremy McKinnon
Thought: We've learned Jeremy McKinnon co-wrote the song Right Back At It Again. Now, we need to find out what city he was born in.
Tool: Search
Tool Input: Where was Jeremy McKinnon born?
Observation: Gainsville, Florida
Thought: We've learned Gainsville, Florida is the city Jeremy McKinnon was born in. Now, we can answer the question.
Final Answer: Gainsville, Florida
##
Question: {query}
Thought:{transcript}
"""

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--search_api_key", default=None, type=str, help="The Serper.dev or SerpAPI key.")
parser.add_argument('--llm_name', choices=['THUDM/chatglm-6b', "THUDM/chatglm-6b-v1.1", "gpt-3.5-turbo", "gpt-4"], default="THUDM/chatglm-6b-v1.1", help="The chatbot models ")
parser.add_argument("--api_key", default=None, type=str, help="The API Key.")
args = parser.parse_args()
# yapf: enable


def search_and_action_example():
    pn = PromptNode(
        args.llm_name,
        max_length=256,
        api_key=args.api_key,
        default_prompt_template="question-answering-with-document-scores",
    )

    # https://serper.dev
    web_retriever = WebRetriever(api_key=args.search_api_key, top_search_results=2)
    pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=pn)

    prompt_node = PromptNode(args.llm_name, api_key=args.api_key, max_length=512, stop_words=["Observation:"])

    web_qa_tool = Tool(
        name="Search",
        pipeline_or_node=pipeline,
        description="useful for when you need to Google questions.",
        output_variable="results",
    )
    few_shot_agent_template = PromptTemplate("few-shot-react", prompt_text=few_shot_prompt)
    # Time to initialize the Agent specifying the PromptNode to use and the Tools
    agent = Agent(
        prompt_node=prompt_node,
        prompt_template=few_shot_agent_template,
        tools_manager=ToolsManager([web_qa_tool]),
        max_steps=8,
    )
    hotpot_questions = [
        "what is the capital of China?",
        "What year was the father of the Princes in the Tower born?",
        "Which author is English: John Braine or Studs Terkel?",
    ]
    for question in hotpot_questions:
        result = agent.run(query=question)
        print(f"\n{result['transcript']}")


if __name__ == "__main__":
    search_and_action_example()
