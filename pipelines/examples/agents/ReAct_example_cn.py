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
你是一个乐于助人、知识渊博的人工智能助手。为了实现正确回答复杂问题的目标，您可以使用以下工具:
搜索: 当你需要用谷歌搜索问题时很有用。你应该问一些有针对性的问题，例如，谁是安东尼·迪雷尔的兄弟？
要回答问题，你需要经历多个步骤，包括逐步思考和选择合适的工具及其输入；工具将以观察作为回应。当您准备好接受最终答案时，回答"最终答案":
示例:
##
问题: 哈利波特的作者是谁？
思考: 让我们一步一步地思考。要回答这个问题，我们首先需要了解哈利波特是什么。
工具: 搜索
工具输入: 哈利波特是什么？
观察: 哈利波特是一系列非常受欢迎的魔幻小说，以及后来的电影和衍生作品。
思考: 我们了解到哈利波特是一系列魔幻小说。现在我们需要找到这些小说的作者是谁。
工具: 搜索
工具输入: 哈利波特的作者是谁？
观察: 哈利波特系列的作者是J.K.罗琳（J.K. Rowling）。
思考: 根据搜索结果，哈利波特系列的作者是J.K.罗琳。所以最终答案是J.K.罗琳。
最终答案: J.K.罗琳
##
问题: {query}
思考:{transcript}
"""

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--search_api_key", default=None, type=str, help="The SerpAPI key.")
parser.add_argument('--llm_name', choices=['ernie-bot', 'THUDM/chatglm-6b', "gpt-3.5-turbo", "gpt-4"], default="THUDM/chatglm-6b", help="The chatbot models ")
parser.add_argument("--api_key", default=None, type=str, help="The API Key.")
parser.add_argument("--secret_key", default=None, type=str, help="The secret key.")
args = parser.parse_args()
# yapf: enable


def search_and_action_example():

    qa_template = PromptTemplate(
        name="文档问答",
        prompt_text="使用以下段落作为来源回答以下问题。"
        "答案应该简短，最多几个字。\n"
        "段落:\n{documents}\n"
        "问题: {query}\n\n"
        "说明: 考虑以上所有段落及其相应的分数，得出答案。 "
        "虽然一个段落可能得分很高， "
        "但重要的是要考虑同一候选答案的所有段落，以便准确回答。\n\n"
        "在考虑了所有的可能性之后，最终答案是:\n",
    )
    pn = PromptNode(
        args.llm_name,
        max_length=512,
        default_prompt_template=qa_template,
        api_key=args.api_key,
        secret_key=args.secret_key,
    )

    # https://serpapi.com/dashboard
    web_retriever = WebRetriever(api_key=args.search_api_key, engine="bing", top_search_results=2)
    pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=pn)

    prompt_node = PromptNode(
        args.llm_name, max_length=512, api_key=args.api_key, secret_key=args.secret_key, stop_words=["观察: "]
    )

    web_qa_tool = Tool(
        name="搜索",
        pipeline_or_node=pipeline,
        description="当你需要用谷歌搜索问题时很有用。",
        output_variable="results",
    )
    few_shot_agent_template = PromptTemplate("few-shot-react", prompt_text=few_shot_prompt)
    # Time to initialize the Agent specifying the PromptNode to use and the Tools
    agent = Agent(
        prompt_node=prompt_node,
        prompt_template=few_shot_agent_template,
        tools_manager=ToolsManager(
            tools=[web_qa_tool],
            tool_pattern=r"工具:\s*(\w+)\s*工具输入:\s*(?:\"([\s\S]*?)\"|((?:.|\n)*))\s*",
        ),
        max_steps=8,
        final_answer_pattern=r"最终答案\s*:\s*(.*)",
    )
    hotpot_questions = [
        " 成龙的儿子几岁了?",
    ]
    for question in hotpot_questions:
        result = agent.run(query=question)
        print(f"\n{result['transcript']}")


if __name__ == "__main__":
    search_and_action_example()
