# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#
# 相关材料：
#   ReAct Prompting 原理简要介绍，不包含代码实现：
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_prompt.md
#   基于 model.chat 接口（对话模式）的 ReAct Prompting 实现（含接入 LangChain 的工具实现）：
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/langchain_tooluse.ipynb
#   基于 model.generate 接口（续写模式）的 ReAct Prompting 实现，比 chat 模式的实现更复杂些：
#       https://github.com/QwenLM/Qwen-7B/blob/main/examples/react_demo.py（本文件）
#
from __future__ import annotations

import json
import os

import json5
import paddle
from function_call.schema import FunctionCallMessage, Message, load_messages_from_file
from function_call.utils import split_turn_messages

from paddlenlp.generation import GenerationConfig
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

# 将一个插件的关键信息拼接成一段文本的模版。
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

# ReAct prompting 的 instruction 模版，将包含插件的详细信息。
PROMPT_REACT = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""


#
# 本示例代码的入口函数。
#
# 输入：
#   prompt: 用户的最新一个问题。
#   history: 用户与模型的对话历史，是一个 list，
#       list 中的每个元素为 {"user": "用户输入", "bot": "模型输出"} 的一轮对话。
#       最新的一轮对话放 list 末尾。不包含最新一个问题。
#   list_of_plugin_info: 候选插件列表，是一个 list，list 中的每个元素为一个插件的关键信息。
#       比如 list_of_plugin_info = [plugin_info_0, plugin_info_1, plugin_info_2]，
#       其中 plugin_info_0, plugin_info_1, plugin_info_2 这几个样例见本文档前文。
#
# 输出：
#   模型对用户最新一个问题的回答。
#


def build_user_bot_tuple_messages(messages: list[Message]) -> list[tuple[str, str]]:
    """build [user, bot] message for qwen inference module"""
    tuple_messages, current_messages = [], []
    for message in messages:
        if current_messages:
            if messages.role == "assistant":
                current_messages.append(message)
                tuple_messages.append(current_messages)
                current_messages = []
        else:
            if messages.role == "user":
                current_messages.append(message)
    if current_messages:
        tuple_messages.append(current_messages)
    return tuple_messages


def eval_function_call(model, tokenizer, file_path: str):
    # convert dataset to messages
    raw_messages = load_messages_from_file(file_path)
    result = []
    for raw_message in raw_messages:
        result.append(eval_function_call_messages(model, tokenizer, raw["messages"], raw["tools"]))
    return result


def eval_function_call_messages(model, tokenizer, messages: list[Message], tools: list[dict]):
    """run with messages which contains multi-turns message"""
    # 0. trans List[Messages] to qwen tuple messages
    assistant_index = 0
    history = []
    text = ""
    # construct tool plugin info
    for tool in tools:
        tool["name_for_model"] = tool["name"]
        tool["name_for_human"] = tool.pop("name")
        tool["description_for_model"] = tool.pop("description")
        tool["args_format"] = "json"

    predicted_messages = []

    # construct history message
    turns_messages = split_turn_messages(messages)
    for user_message, function_call_messages, assistant_message in turns_messages[:-1]:
        tool_content = ""
        for tool_request_message, tool_response_message in function_call_messages:
            tool_content += f"Thought: {tool_request_message.function_call['thoughts']}\n"
            tool_content += f"Action: {tool_request_message.function_call['name']}\n"
            tool_content += f"Action Input: {tool_request_message.function_call['parameters']}\n"
            tool_content += "Observation: tool_response_message['content']\n"
        tool_content += "Final Answer: " + assistant_message.content
        history.append((user_message.content, tool_content))

    user_message, function_call_messages, assistant_message = turns_messages[-1]
    history.append((user_message.content, ""))

    planning_prompt = build_input_text(history, tools)
    # output = steps_text_completion(model, tokenizer, planning_prompt, stop_words=['Observation:', 'Observation:\n'])
    output = text_completion(model, tokenizer, planning_prompt, stop_words=["Observation:", "Observation:\n"])
    thoughts, action, action_input, output = parse_latest_plugin_call(output)
    return Message.from_dict(
        role="assistant", function_call=dict(name=action, parameters=action_input, thoughts=thoughts)
    )


def llm_with_plugin(model, tokenizer, prompt: str, history: list[Message], list_of_plugin_info=()):
    chat_history = [(x["user"], x["bot"]) for x in history] + [(prompt, "")]

    # 需要让模型进行续写的初始文本
    planning_prompt = build_input_text(chat_history, list_of_plugin_info)

    text = ""
    while True:
        output = text_completion(
            model, tokenizer, planning_prompt + text, stop_words=["Observation:", "Observation:\n"]
        )
        thought, action, action_input, output = parse_latest_plugin_call(output)
        if action:  # 需要调用插件
            # action、action_input 分别为需要调用的插件代号、输入参数
            # observation是插件返回的结果，为字符串
            observation = call_plugin(action, action_input)
            output += f"\nObservation: {observation}\nThought:"
            text += output
        else:  # 生成结束，并且不再需要调用插件
            text += output
            break

    new_history = []
    new_history.extend(history)
    new_history.append({"user": prompt, "bot": text})
    return text, new_history


# 将对话历史、插件信息聚合成一段初始文本
def build_input_text(chat_history, list_of_plugin_info) -> str:
    # 候选插件的详细信息
    tools_text = []
    for plugin_info in list_of_plugin_info:
        tool = TOOL_DESC.format(
            name_for_model=plugin_info["name_for_model"],
            name_for_human=plugin_info["name_for_human"],
            description_for_model=plugin_info["description_for_model"],
            parameters=json.dumps(plugin_info["parameters"], ensure_ascii=False),
        )
        if plugin_info.get("args_format", "json") == "json":
            tool += " Format the arguments as a JSON object."
        elif plugin_info["args_format"] == "code":
            tool += " Enclose the code within triple backticks (`) at the beginning and end of the code."
        else:
            raise NotImplementedError
        tools_text.append(tool)
    tools_text = "\n\n".join(tools_text)

    # 候选插件的代号
    tools_name_text = ", ".join([plugin_info["name_for_model"] for plugin_info in list_of_plugin_info])

    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
    for i, (query, response) in enumerate(chat_history):
        if list_of_plugin_info:  # 如果有候选插件
            # 倒数第一轮或倒数第二轮对话填入详细的插件信息，但具体什么位置填可以自行判断
            if (len(chat_history) == 1) or (i == len(chat_history) - 2):
                query = PROMPT_REACT.format(
                    tools_text=tools_text,
                    tools_name_text=tools_name_text,
                    query=query,
                )
        query = query.lstrip("\n").rstrip()  # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        response = response.lstrip("\n").rstrip()  # 重要！若不 strip 会与训练时数据的构造方式产生差异。
        # 使用续写模式（text completion）时，需要用如下格式区分用户和AI：
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"

    assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
    prompt = prompt[: -len(f"{im_end}")]
    return prompt


def steps_text_completion(model, tokenizer, input_text: str, stop_words, schema: dict) -> str:  # 作为一个文本续写模型来使用
    im_end = "<|im_end|>"
    if im_end not in stop_words:
        stop_words = stop_words + [im_end]
    stop_words_ids = [tokenizer.encode(w)["input_ids"] for w in stop_words]

    # TODO: 增加流式输出的样例实现
    inputs = {key: paddle.to_tensor([value], dtype="int64") for key, value in tokenizer(input_text).items()}
    import pdb

    pdb.set_trace()
    output = model.generate(**inputs, decode_strategy="greedy_search", eos_token_id=stop_words_ids)[0]
    output = output.tolist()[0]
    output = tokenizer.decode(output, errors="ignore")
    output = input_text + output
    assert output.startswith(input_text)
    output = output[len(input_text) :].replace("<|endoftext|>", "").replace(im_end, "")

    for stop_str in stop_words:
        idx = output.find(stop_str)
        if idx != -1:
            output = output[: idx + len(stop_str)]
    return output  # 续写 input_text 的结果，不包含 input_text 的内容


def text_completion(model, tokenizer, input_text: str, stop_words) -> str:  # 作为一个文本续写模型来使用
    im_end = "<|im_end|>"
    if im_end not in stop_words:
        stop_words = stop_words + [im_end]
    stop_words_ids = [tokenizer.encode(w)["input_ids"] for w in stop_words]

    # TODO: 增加流式输出的样例实现
    inputs = {key: paddle.to_tensor([value], dtype="int64") for key, value in tokenizer(input_text).items()}
    output = model.generate(**inputs, decode_strategy="greedy_search", eos_token_id=stop_words_ids)[0]
    output = output.tolist()[0]
    output = tokenizer.decode(output, errors="ignore")
    output = input_text + output
    assert output.startswith(input_text)
    output = output[len(input_text) :].replace("<|endoftext|>", "").replace(im_end, "")

    for stop_str in stop_words:
        idx = output.find(stop_str)
        if idx != -1:
            output = output[: idx + len(stop_str)]
    return output  # 续写 input_text 的结果，不包含 input_text 的内容


def parse_latest_plugin_call(text):
    plugin_name, plugin_args = "", ""
    i = text.rfind("\nAction:")
    j = text.rfind("\nAction Input:")
    k = text.rfind("\nObservation:")
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + "\nObservation:"  # Add it back.
        k = text.rfind("\nObservation:")
        plugin_name = text[i + len("\nAction:") : j].strip()
        plugin_args = text[j + len("\nAction Input:") : k].strip()
        text = text[:k]
    thought = text[text.find("Thought: ") + len("Thought: ") : i]
    return thought, plugin_name, plugin_args, text


#
# 输入：
#   plugin_name: 需要调用的插件代号，对应 name_for_model。
#   plugin_args：插件的输入参数，是一个 dict，dict 的 key、value 分别为参数名、参数值。
# 输出：
#   插件的返回结果，需要是字符串。
#   即使原本是 JSON 输出，也请 json.dumps(..., ensure_ascii=False) 成字符串。
#
def call_plugin(plugin_name: str, plugin_args: str) -> str:
    #
    # 请开发者自行完善这部分内容。这里的参考实现仅是 demo 用途，非生产用途。
    #
    if plugin_name == "google_search":
        # 使用 SerpAPI 需要在这里填入您的 SERPAPI_API_KEY！
        os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY", default="")
        from langchain import SerpAPIWrapper

        return SerpAPIWrapper().run(json5.loads(plugin_args)["search_query"])
    elif plugin_name == "image_gen":
        import urllib.parse

        prompt = json5.loads(plugin_args)["prompt"]
        prompt = urllib.parse.quote(prompt)
        return json.dumps({"image_url": f"https://image.pollinations.ai/prompt/{prompt}"}, ensure_ascii=False)
    else:
        raise NotImplementedError


def test(model):
    tools = [
        {
            "name_for_human": "谷歌搜索",
            "name_for_model": "google_search",
            "description_for_model": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。",
            "parameters": [
                {
                    "name": "search_query",
                    "description": "搜索关键词或短语",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        {
            "name_for_human": "文生图",
            "name_for_model": "image_gen",
            "description_for_model": "文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL",
            "parameters": [
                {
                    "name": "prompt",
                    "description": "英文关键词，描述了希望图像具有什么内容",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
    ]
    history = []
    for query in ["你好", "搜索一下谁是周杰伦", "再搜下他老婆是谁", "给我画个可爱的小猫吧，最好是黑猫"]:
        print(f"User's Query:\n{query}\n")
        response, history = llm_with_plugin(model, tokenizer, prompt=query, history=history, list_of_plugin_info=tools)
        print(f"Qwen's Response:\n{response}\n")


if __name__ == "__main__":
    name = "qwen/qwen-7b-chat"
    name = "qwen/qwen-1_8b-chat"
    tokenizer = AutoTokenizer.from_pretrained(name)
    generation_config = GenerationConfig.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
    )
    model.eval()
    model.generation_config = generation_config
    model.generation_config.top_k = 1
    test(model, tokenizer)
