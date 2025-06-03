# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Optional, Union

from typing_extensions import override

from .tool_utils import FunctionCall, get_tool_utils

SLOTS = list[Union[str, set[str], dict[str, str]]]


KG_RES_MARKUPS = [
    "[<kg-res>]",
    "[</kg-res>]",
    "[<kg-yes>]",
    "[</kg-yes>]",
    "[<kg-cs-yes>]",
    "[</kg-cs-yes>]",
    "[<kg-cs-no>]",
    "[</kg-cs-no>]",
]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTIONCALL = "function_call"
    TOOLS = "tools"
    OBSERVATION = "observation"


def extract_knowledge(text):
    """Extracts structured knowledge from text markup.

    Args:
        text (str): Input text containing markup.

    Returns:
        str: Processed knowledge string.

    Raises:
        ValueError: If no valid knowledge pattern found.
    """

    if any(markup in text for markup in KG_RES_MARKUPS):
        for markup in KG_RES_MARKUPS + ["[<image>]", "[</image>]"]:
            text = text.replace(markup, "")
        text = f"知识库：{text.strip()}\n根据所提供的知识库信息，回答问题并补全对话："
        return text

    res = re.findall(
        r"\[<search-res>\](.*?)\[<\/search-res>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = f"{text.strip()}\n根据以上参考文章回答问题，补全对话"
        return text

    res = re.findall(
        r"\[<prompt-res>\](.*?)\[<\/prompt-res>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = text.strip()
        return text

    res = re.findall(
        r"\[<compute-res>\](.*?)\[<\/compute-res>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = f"参考文章1：{text.strip()}\n根据以上参考文章回答问题，补全对话"
        return text

    res = re.findall(
        r"\[<citation-ref>\](.*?)\[<\/citation-ref>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = (
            "请参考搜索结果回答下面问题并使用引用标记来标注回答内容参考的搜索结果序号，"
            "例如^[1]^ (引用单个搜索结果）,^[1][2]^（引用多个搜索结果），"
            "其中方括号中的数字是搜索结果序号。引用标记只能出现在句尾标点符号前。\n"
            "以下是搜索结果（每行开头[1]、[2]、...是搜索结果序号），"
            f"可以对答案中的核心部分进行markdown加粗（**加粗内容**）：\n{text.strip()}\n"
            "根据以上搜索结果回答问题并标注引用，补全对话"
        )
        return text

    res = re.findall(
        r"\[<retrieve-ref>\](.*?)\[<\/retrieve-ref>\]",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if len(res) > 0:
        text = res[0]
        text = (
            "请你扮演一个专家，参考搜索结果中正确、可信、高质量的信息回答问题，并注明答案中引用的搜索结果，"
            "格式为^[2]^表示引用了第2条搜索结果，^[1][3]^表示引用第1和第3条搜索结果。"
            "每条搜索结果包含若干相关内容片段。同时你需要遵循以下原则回答问题：\n"
            "1. 严格遵循搜索结果作答，可以承认不知道答案，并尝试给出一些搜索结果中的相关背景信息。\n"
            "2. 如果搜索结果存在多种可能的答案，要罗列出每种情况。\n"
            "3. 如果问题涉及金融、医疗、法律等存在风险的领域，请在结尾提醒用户注意并进行免责说明。\n"
            f"搜索结果：\n{text.strip()}\n\n现在，请根据上面的搜索结果回答问题并标注引用，补全对话"
        )
        return text

    raise ValueError(f"Cannot extract knowledge from `{text}`")


@dataclass
class Formatter(ABC):
    slots: SLOTS = field(default_factory=list)
    tool_format: Optional[str] = None

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS:
        r"""Forms a list of slots according to the inputs to encode."""
        ...


@dataclass
class EmptyFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        return self.slots


@dataclass
class StringFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError(f"Expected a string, got {name} : s{value}")

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError(f"Input must be string, set[str] or dict[str, str], got {type(slot)}.")
        return elements


@dataclass
class KnowledgeFormatter(StringFormatter):
    @override
    def apply(self, **kwargs) -> SLOTS:
        content: str = extract_knowledge(kwargs.pop("content")) + "\n"
        idx: int = kwargs.pop("idx")
        return super().apply(content=content, idx=idx)


@dataclass
class FunctionFormatter(StringFormatter):
    def __post_init__(self):
        super().__post_init__()
        self.tool_utils = get_tool_utils(self.tool_format)

    @override
    def apply(self, **kwargs) -> SLOTS:
        content: str = kwargs.pop("content")
        functions: list[FunctionCall] = []
        try:
            tool_calls = json.loads(content)
            if not isinstance(tool_calls, list):  # parallel function call
                tool_calls = [tool_calls]

            for tool_call in tool_calls:
                functions.append(
                    FunctionCall(tool_call["name"], json.dumps(tool_call["arguments"], ensure_ascii=False))
                )

        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in function message: {str([content])}.")  # flat string

        function_str = self.tool_utils.function_formatter(functions)

        return super().apply(content=function_str)


@dataclass
class ToolFormatter(Formatter):
    def __post_init__(self):
        self.tool_utils = get_tool_utils(self.tool_format)

    @override
    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        try:
            tools = json.loads(content)
            return self.tool_utils.tool_formatter(tools) if len(tools) != 0 else ""
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in tool description: {str([content])}.")  # flat string

    @override
    def extract(self, content: str) -> Union[str, list["FunctionCall"]]:
        return self.tool_utils.tool_extractor(content)
