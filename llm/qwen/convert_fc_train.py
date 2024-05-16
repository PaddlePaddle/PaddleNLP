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

from __future__ import annotations

import json
import sys

from function_call.schema import Message, group_round_messages, parse_messages

# the following code is copied from: https://github.com/QwenLM/Qwen/blob/main/examples/function_call_finetune_examples.py#L42
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

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

Begin!"""


def build_react_instruction(functions):
    if isinstance(functions[0], str):
        return "\n".join(functions)

    tools_text = []
    tools_name_text = []
    for func_info in functions:
        name = func_info.get("name", "")
        name_m = func_info.get("name_for_model", name)
        name_h = func_info.get("name_for_human", name)
        desc = func_info.get("description", "")
        desc_m = func_info.get("description_for_model", desc)
        tool = TOOL_DESC.format(
            name_for_model=name_m,
            name_for_human=name_h,
            description_for_model=desc_m,
            parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
        )
        tools_text.append(tool)
        tools_name_text.append(name_m)
    tools_text = "\n\n".join(tools_text)
    tools_name_text = ", ".join(tools_name_text)
    instruction = REACT_INSTRUCTION.format(
        tools_text=tools_text,
        tools_name_text=tools_name_text,
    )
    return instruction


def construct_system_message(tools: list[dict]):
    # refer to: https://github.com/QwenLM/Qwen/blob/main/examples/function_call_finetune_examples.py#L42
    pass


def convert(file: str) -> list[Message]:
    # 0. load files
    all_data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            data["messages"] = parse_messages(data.pop("messages"))
            all_data.append(data)

    result = []
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    nl_token = "\n"
    _system = "system" + nl_token
    _user = "user" + nl_token
    _assistant = "assistant" + nl_token

    # refer to:
    # 1. construct tgt & src
    all_train_dataset = []
    for data in all_data:
        rounds, learnable = [], []
        # construct system message
        instruction = build_react_instruction(data["tools"])
        function_call_content = ""
        for index, message in enumerate(data["messages"]):
            role = message.role
            if index == 0:
                assert role in ["user", "system"]
                if role == "system":
                    rounds += [im_start] + [_system + message.content + "\n" + instruction] + [im_end]
                    learnable += [True, False, True]
                else:
                    rounds += (
                        [im_start]
                        + [_system + instruction]
                        + [im_end + im_start]
                        + [_user + message.content]
                        + [im_end]
                    )
                    learnable += [True, False, True, False, True]
                continue
            if role == "user":
                rounds += [im_start] + [_user + "Question:" + message.content] + [im_end]
                learnable += [True, False, True]
                function_call_content = ""
            elif role == "assistant":
                if message.is_function_call_message:
                    function_call = message.function_call
                    content = f"Thought: {function_call.thoughts}\n"
                    content += f"Action: {function_call.name}\n"
                    content += f"Action Input: {function_call.parameters}\nObservation: "
                    rounds += [content]
                    learnable += [True]
                else:
                    if not function_call_content:
                        rounds += [message.content + im_end]
                        learnable += [True]
                    else:
                        function_call_content += [
                            "Thought: I now know the final answer\nFinal Answer: " + messages.content + im_end
                        ]
                        learnable += [True]
                        function_call_content = ""
            elif role == "function":
                rounds += [message.content] + [nl_token]
                learnable += [False, True]

        if len(learnable) == 0:
            continue

        assert len(rounds) == len(learnable)
        if index < 10:
            print("\n\n===============Print the Rounds Info===============")
            print("".join(rounds))

        # group rounds and learnables
        src, tgt = [""], []
        temp = []
        pre_stage = learnable[0]
        for i in range(len(learnable)):
            if learnable[i] != pre_stage:
                if pre_stage == True:
                    tgt.append("".join(temp))
                else:
                    src.append("".join(temp))
                pre_stage = not pre_stage
                temp = [rounds[i]]
            else:
                temp.append(rounds[i])
        if pre_stage == True:
            tgt.append("".join(temp))
        else:
            src.append("".join(temp))
        assert len(src) == len(tgt)
        all_train_dataset.append((src, tgt))

    return all_train_dataset


def convert_train_files(source_file: str, target_file: str):
    dataset = convert(source_file)
    with open(target_file, "w+", encoding="utf-8") as f:
        for src, tgt in dataset:
            f.write(json.dumps(dict(src=src, tgt=tgt), ensure_ascii=False) + "\n")


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(role, allowed_special=set()) + nl_tokens + tokenizer.encode(
                content, allowed_special=set()
            )

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"

            current_context_size = len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


def eval_function_call(model, eval_file_path: str) -> Dict[str, float]:
    file_path = os.path.join(eval_file_path, "eval.json")
    raw_messages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "function_call" in data:
                raw_messages.append(FunctionCallMessage(**data))
            else:
                raw_messages.append(Message(**data))

    # 0. generate messages
    # for messages in raw_messages:
    # 1. eval the messages


if __name__ == "__main__":
    convert_train_files(sys.argv[1], sys.argv[2])
