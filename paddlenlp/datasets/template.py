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

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .formatter import EmptyFormatter, KnowledgeFormatter, Role, StringFormatter

if TYPE_CHECKING:
    from paddlenlp.transformers import PretrainedTokenizer

    from .formatter import SLOTS, Formatter


logger = logging.getLogger(__name__)


def contain_tokens(text, token_list):
    """Checks if any token in list exist in the text.

    Args:
        text (List[str]): Input text sequences to check.
        token_list (List[str]): tokens to search for.

    Returns:
        bool: True if any is found, False otherwise.
    """

    for sp_token in token_list:
        for x in text:
            if sp_token in x:
                return True
    return False


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_knowledge: "Formatter"
    format_tools: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: list[str]
    efficient_eos: bool
    replace_eos: bool
    thought_words: tuple
    replace_jinja_template: bool

    def encode_oneturn(
        self,
        tokenizer: "PretrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> tuple[list[int], list[int]]:
        r"""Return a single pair of token ids representing prompt and response respectively."""
        system = None
        if messages[0]["role"] == Role.SYSTEM.value:
            system = messages[0]["content"]
            messages = messages[1:]
        encoded_messages = self._encode(tokenizer, messages, system)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        response_ids = encoded_messages[-1]
        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: "PretrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        system = None
        if messages[0]["role"] == Role.SYSTEM.value:
            system = messages[0]["content"]
            messages = messages[1:]
        encoded_messages = self._encode(tokenizer, messages, system)
        return [
            (encoded_messages[i], encoded_messages[i + 1] if i + 1 < len(encoded_messages) else None)
            for i in range(0, len(encoded_messages), 2)
        ]

    def _convert_elements_to_ids(self, tokenizer: "PretrainedTokenizer", elements: "SLOTS") -> list[int]:
        r"""Convert elements to token ids."""
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)["input_ids"]
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(f"Input must be string, set[str] or dict[str, str], got {type(elem)}")

        return token_ids

    def _encode(
        self,
        tokenizer: "PretrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
    ) -> list[list[int]]:
        r"""Encode formatted inputs to pairs of token ids.

        Turn 0: prefix + system + query        resp
        Turn t: query                          resp.
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system:
                    elements += self.format_system.apply(content=(system))

            if message["role"] == Role.USER:
                if (
                    self.format_knowledge
                    and hasattr(tokenizer, "markup_tokens")
                    and i == len(messages) - 2
                    and contain_tokens([message["content"]], tokenizer.markup_tokens)
                ):
                    elements += self.format_knowledge.apply(content=message["content"], idx=str(i // 2))
                else:
                    elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.FUNCTIONCALL:
                elements += self.format_function.apply(content=message["content"])
            elif message["role"] == Role.TOOLS:
                elements += self.format_tools.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            if len(elements) > 0:
                encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    @staticmethod
    def _add_or_replace_eos_token(tokenizer: "PretrainedTokenizer", eos_token: str) -> None:
        r"""Add or replace eos token to the tokenizer."""
        if tokenizer.eos_token == eos_token:
            return

        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

        if is_added:
            logger.info(f"Add eos token: {tokenizer.eos_token}.")
        else:
            logger.info(f"Replace eos token: {tokenizer.eos_token}.")

        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    def fix_special_tokens(self, tokenizer: "PretrainedTokenizer") -> None:
        r"""Add eos token and pad token to the tokenizer."""
        stop_words = self.stop_words
        if self.replace_eos:
            if not stop_words:
                raise ValueError("Stop words are required to replace the EOS token.")

            self._add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
            stop_words = stop_words[1:]

        if tokenizer.eos_token_id is None:
            self._add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Add pad token: {tokenizer.pad_token}")

        if stop_words:
            num_added_tokens = tokenizer.add_special_tokens(
                dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
            )
            logger.info("Add {} to stop words.".format(",".join(stop_words)))
            if num_added_tokens > 0:
                logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    @staticmethod
    def _jinja_escape(content: str) -> str:
        r"""Escape single quotes in content."""
        return content.replace("'", r"\'")

    @staticmethod
    def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PretrainedTokenizer", placeholder: str = "content") -> str:
        r"""Convert slots to jinja template."""
        slot_items = []
        for slot in slots:
            if isinstance(slot, str):
                slot_pieces = slot.split("{{content}}")
                if slot_pieces[0]:
                    slot_items.append("'" + Template._jinja_escape(slot_pieces[0]) + "'")
                if len(slot_pieces) > 1:
                    slot_items.append(placeholder)
                    if slot_pieces[1]:
                        slot_items.append("'" + Template._jinja_escape(slot_pieces[1]) + "'")
            elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
                if "bos_token" in slot and tokenizer.bos_token_id is not None:
                    slot_items.append("'" + tokenizer.bos_token + "'")
                elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                    slot_items.append("'" + tokenizer.eos_token + "'")
            elif isinstance(slot, dict):
                slot_items.append("'" + slot.get("token") + "'")

        return " + ".join(slot_items)

    def _get_jinja_template(self, tokenizer: "PretrainedTokenizer") -> str:
        r"""Return the jinja template."""
        prefix = self._convert_slots_to_jinja(self.format_prefix.apply(), tokenizer)
        system = self._convert_slots_to_jinja(self.format_system.apply(), tokenizer, placeholder="system_message")
        user = self._convert_slots_to_jinja(self.format_user.apply(), tokenizer)
        assistant = self._convert_slots_to_jinja(self.format_assistant.apply(), tokenizer)
        function_call = self._convert_slots_to_jinja(self.format_function.apply(), tokenizer)
        tools = self._convert_slots_to_jinja(self.format_tools.apply(), tokenizer)
        observation = self._convert_slots_to_jinja(self.format_observation.apply(), tokenizer)
        jinja_template = ""
        if prefix:
            jinja_template += "{{ " + prefix + " }}"

        if self.default_system:
            jinja_template += "{% set system_message = '" + self._jinja_escape(self.default_system) + "' %}"

        if not self.format_knowledge:
            jinja_template += (
                "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
                "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
                "{% if system_message is defined %}{{ " + system + " }}{% endif %}"
                "{% for message in loop_messages %}"
                "{% set content = message['content'] %}"
                "{% if message['role'] == 'user' %}"
                "{{ " + user + " }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ " + assistant + " }}"
                "{% elif message['role'] == 'function_call' %}"
                "{{ " + function_call + " }}"
                "{% elif message['role'] == 'tools' %}"
                "{{ " + tools + " }}"
                "{% elif message['role'] == 'observation' %}"
                "{{ " + observation + " }}"
                "{% endif %}"
                "{% endfor %}"
            )
        else:
            jinja_template += (
                "{% set KG_RES_MARKUPS = ['[<kg>]', '[</kg>]', '[<kg-raw>]', '[</kg-raw>]'] %}{{'<|begin_of_sentence|>'}}"
                "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}"
                "{% else %}{% set loop_messages = messages %}{% endif %}{% if system_message is defined %}{{ system_message + '\n' }}"
                "{% endif %}{% set ns = namespace(last_user_message=None) %}{% for message in loop_messages %}{% if message['role'] == 'user' %}"
                "{% set ns.last_user_message = message['content'] %}{% endif %}{% endfor %}{% for message in loop_messages %}"
                "{% set content = message['content'] %}{% if message['role'] == 'user' %}{% if content == ns.last_user_message %}{% set text = content %}"
                "{% set  ns = namespace(has_markup=False) %}{% for markup in KG_RES_MARKUPS + ['[<image>]', '[</image>]'] %}{% if markup in text %}"
                "{% set ns.has_markup = True %}{% set text = text.replace(markup, '') %}{% endif %}{% endfor %}{% if ns.has_markup == True %}"
                "{{ 'User: 知识库：' + text.strip() + '\n根据所提供的知识库信息，回答问题并补全对话：\nAssistant: ' }}{% else %}"
                "{% set res = text | regex_findall('\[<search-res>\](.*?)\[</search-res>\]', multiline=True, dotall=True) %}{% if res %}"
                "{{ 'User: ' + res[0].strip() + '\n根据以上参考文章回答问题，补全对话\nAssistant: ' }}{% else %}"
                "{% set res = text | regex_findall('\[<prompt-res>\](.*?)\[</prompt-res>\]', multiline=True, dotall=True) %}{% if res %}"
                "{{ 'User: ' + res[0].strip() + '\nAssistant: ' }}{% else %}"
                "{% set res = text | regex_findall('\[<compute-res>\](.*?)\[</compute-res>\]', multiline=True, dotall=True) %}{% if res %}"
                "{{ 'User: 参考文章1：' + res[0].strip() + '\n根据以上参考文章回答问题，补全对话\nAssistant: ' }}{% else %}"
                "{% set res = text | regex_findall('\[<citation-ref>\](.*?)\[</citation-ref>\]', multiline=True, dotall=True) %}"
                "{% if res %} User: 请参考搜索结果回答下面问题并使用引用标记来标注回答内容参考的搜索结果序号，例如^[1]^ (引用单个搜索结果）,^[1][2]^（引用多个搜索结果），其中方括号中的数字是搜索结果序号。引用标记只能出现在句尾标点符号前。 以下是搜索结果（每行开头[1]、[2]、...是搜索结果序号），可以对答案中的核心部分进行markdown加粗（加粗内容）： {{ res[0].strip() }} 根据以上搜索结果回答问题并标注引用，补全对话 Assistant: {% else %}"
                "{% set res = text | regex_findall('\[<retrieve-ref>\](.*?)\[</retrieve-ref>\]', multiline=True, dotall=True) %}"
                "{% if res %} User: 请你扮演一个专家，参考搜索结果中正确、可信、高质量的信息回答问题，并注明答案中引用的搜索结果，格式为^[2]^表示引用了第2条搜索结果，^[1][3]^表示引用第1和第3条搜索结果。每条搜索结果包含若干相关内容片段。同时你需要遵循以下原则回答问题： 1. 严格遵循搜索结果作答，可以承认不知道答案，并尝试给出一些搜索结果中的相关背景信息。 2. 如果搜索结果存在多种可能的答案，要罗列出每种情况。 3. 如果问题涉及金融、医疗、法律等存在风险的领域，请在结尾提醒用户注意并进行免责说明。 搜索结果： {{ res[0].strip() }} 现在，请根据上面的搜索结果回答问题并标注引用，补全对话 Assistant: {% else %}"
                "{{ 'User: ' + content + '\nAssistant: ' }}{% endif %}{% endif %}{% endif %}{% endif %}{% endif %}{% endif %}"
                "{% else %}{{ 'User: ' + content + '\nAssistant: ' }}{% endif %}"
                "{% elif message['role'] == 'assistant' %}{{ content + '<|end_of_sentence|>' }}{% endif %}{% endfor %}"
            )
        return jinja_template

    def fix_jinja_template(self, tokenizer: "PretrainedTokenizer") -> None:
        r"""Replace the jinja template in the tokenizer."""
        if tokenizer.chat_template is None or self.replace_jinja_template:
            try:
                tokenizer.chat_template = self._get_jinja_template(tokenizer)
            except ValueError as e:
                logger.info(f"Cannot add this chat template to tokenizer: {e}.")


TEMPLATES: dict[str, "Template"] = {}


def register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_knowledge: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: Optional[list[str]] = None,
    efficient_eos: bool = False,
    thought_words: Optional[tuple[str, str]] = None,
    replace_eos: bool = False,
    replace_jinja_template: bool = False,
    template_class: type["Template"] = Template,
) -> None:
    r"""Register a chat template.

    To add the following chat template:
    ```
    <s><user>user prompt here
    <model>model response here</s>
    <user>user prompt here
    <model>model response here</s>
    ```

    The corresponding code should be:
    ```
    register_template(
        name="custom",
        format_user=StringFormatter(slots=["<user>{{content}}\n<model>"]),
        format_assistant=StringFormatter(slots=["{{content}}</s>\n"]),
        format_prefix=EmptyFormatter("<s>"),
    )
    ```
    """
    if name in TEMPLATES:
        raise ValueError(f"Template {name} already exists.")

    default_slots = ["{{content}}"] if efficient_eos else ["{{content}}", {"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=default_slots)
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_knowledge=format_knowledge,
        format_prefix=format_prefix or default_prefix_formatter,
        format_function=format_function or default_prefix_formatter,
        format_tools=format_tools or default_prefix_formatter,
        format_observation=format_observation or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words or [],
        efficient_eos=efficient_eos,
        thought_words=thought_words,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
    )


def parse_template(tokenizer: "PretrainedTokenizer") -> "Template":
    r"""Extract a chat template from the tokenizer."""

    def find_diff(short_str: str, long_str: str) -> str:
        i, j = 0, 0
        diff = ""
        while i < len(short_str) and j < len(long_str):
            if short_str[i] == long_str[j]:
                i += 1
                j += 1
            else:
                diff += long_str[j]
                j += 1

        return diff

    """
    1. prefix
    2. system
    2.1. global setting
    2.2. tool des
    3. role USER
    4. role ASSISTANT
    5. role FUNC
    6. role OBSER
    7. THINK option

    {
        messages: [
            {"role": "system", "content": "你的名字是BookWiseBot，是一个专为二手书交易平台服务的智能图书顾问"},
            {"role": "tool", "content": "[{
                "type": "function",
                "name": "search_books",
                "description": "搜索二手书售卖信息，支持按标题、作者、ISBN和出版年份搜索相关结果",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "书籍的标题"},
                        "author": {"type": "string", "description": "书籍的作者"},
                        "isbn": {"type": "string", "description": "书籍的国际标准书号(ISBN)"},
                        "publish_year": {"type": "string","description": "书籍的出版年份"}
                    },
                    "required": []
                },
                "strict": true
            }]",
            {"role": "user", "content": "看看《百年孤独》二手多少钱？"},
            {"role": "function_call, "content": "{'name': 'search_books.call', 'arguments': {"title": "百年孤独"}}"},

            {"role": "observation", "content": "{"books": [{"purchase_link": "http://bookstore.example.com/bookid-12345", "price": "￥80.00", "condition": 95, "book_id": "bookid-12345"}, {"purchase_link": "http://bookstore.example.com/bookid-12346", "price": "￥120.00", "condition": 100, "book_id": "bookid-12346"}, {"purchase_link": "http://usedbooks.example.com/bookid-12347", "price": "￥50.00", "condition": 80, "book_id": "bookid-12347"}]}"},

            {"role": "assistant", "content": "搜集到了三个不同条件《百年孤独》的售价和平均价格为65元。"},
        ],
        think: True,
        global_setting: str
    }
    """
    prefix = tokenizer.decode(tokenizer.encode("")["input_ids"])

    messages = [{"role": "system", "content": "{{content}}"}]
    system_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[len(prefix) :]

    messages = [{"role": "tool", "content": "{{content}}"}]
    tool_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[len(prefix) :]

    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "{{content}}"}]
    user_slot_empty_system = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot_empty_system = user_slot_empty_system[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}]
    user_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot = user_slot[len(prefix) :]

    messages = [{"role": "function_call", "content": "{{content}}"}]
    fc_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    fc_slot = fc_slot[len(prefix) :]

    messages = [{"role": "observation", "content": "{{content}}"}]
    ob_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ob_slot = fc_slot[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}, {"role": "assistant", "content": "{{content}}"}]
    assistant_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    assistant_slot = tokenizer.encode(assistant_slot[len(prefix) + len(user_slot) :], add_special_tokens=False)[
        "input_ids"
    ]
    messages = [
        {"role": "user", "content": "{{content}}"},
        {"role": "assistant", "content": "{{content}}"},
        {"role": "user", "content": "{{content}}"},
    ]

    # In case of <diag_eos_token> + <eos_token>
    assistant_slot_further = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    assistant_slot_further = tokenizer.encode(
        assistant_slot_further[len(prefix) + len(user_slot) :], add_special_tokens=False
    )["input_ids"]

    assistant_slot = tokenizer.decode(os.path.commonprefix([assistant_slot, assistant_slot_further]))

    if len(user_slot) > len(user_slot_empty_system):
        default_system = find_diff(user_slot_empty_system, user_slot)
        sole_system = system_slot.replace("{{content}}", default_system, 1)
        user_slot = user_slot[len(sole_system) :]
    else:  # if defaut_system is empty, user_slot_empty_system will be longer than user_slot
        default_system = ""

    import pdb

    pdb.set_trace()
    return Template(
        format_user=StringFormatter(slots=[user_slot]),
        format_assistant=StringFormatter(slots=[assistant_slot]),
        format_system=StringFormatter(slots=[system_slot]),
        format_knowledge=KnowledgeFormatter(slots=[user_slot]),
        format_prefix=EmptyFormatter(slots=[prefix]) if prefix else EmptyFormatter(),
        format_function=StringFormatter(slots=[fc_slot]),
        format_observation=StringFormatter(slots=[ob_slot]),
        format_tools=StringFormatter(slots=[tool_slot]),
        default_system=default_system,
        # thought_words=thought_words,
        stop_words=[],
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
    )


def get_template_and_fix_tokenizer(tokenizer: "PretrainedTokenizer", template: str = None) -> "Template":
    r"""Get chat template and fixes the tokenizer."""
    if template is None:
        if isinstance(tokenizer.chat_template1, str):
            logger.warning("`template` was not specified, try parsing the chat template from the tokenizer.")
            template = parse_template(tokenizer)
        else:
            logger.warning("`template` was not specified, use `empty` template.")
            template = TEMPLATES["empty"]  # placeholder
    else:
        if template not in TEMPLATES:
            raise ValueError(f"Template {template} does not exist.")

        template = TEMPLATES[template]

    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)
    return template


"""
{% if not add_generation_prompt is defined %}
{% set add_generation_prompt = false %}
{% endif %}
{% set loop_messages = messages %}
{% for message      in loop_messages %}
{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}
{% if loop.index0      == 0 %}
{% set content = bos_token + content %}
{% endif %}
{{ content }}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|start_header_id|>assistant<|end_header_id|>\n     \n' }}
{% else %}
{{ eos_token }}
{% endif %}
Template(efficient_eos=False, replace_eos=False, replace_jinja_template=False)
"""
register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot_id|><|end_of_text|>"]),
    format_system=StringFormatter(
        slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|><|end_of_text|>"]
    ),
    format_prefix=EmptyFormatter(slots=["<|begin_of_text|>"]),
    replace_jinja_template=True,
)


register_template(
    name="aquila",
    format_user=StringFormatter(slots=["Human: {{content}}###Assistant:"]),
    format_assistant=StringFormatter(slots=["{{content}}###"]),
    format_system=StringFormatter(slots=["System: {{content}}###"]),
    default_system=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    stop_words=["</s>"],
)


register_template(
    name="atom",
    format_user=StringFormatter(
        slots=[{"bos_token"}, "Human: {{content}}\n", {"eos_token"}, {"bos_token"}, "Assistant:"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}\n", {"eos_token"}]),
)


register_template(
    name="baichuan",
    format_user=StringFormatter(slots=[{"token": "<reserved_102>"}, "{{content}}", {"token": "<reserved_103>"}]),
    efficient_eos=True,
)


register_template(
    name="45t",
    format_user=StringFormatter(slots=["User: ", "{{content}}\nAssistant: "]),
    format_assistant=StringFormatter(slots=["{{content}}", {"token": "<|end_of_sentence|>"}]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=[{"token": "<|begin_of_sentence|>"}]),
    format_knowledge=KnowledgeFormatter(slots=["User: {{content}}\nAssistant: "]),
    replace_jinja_template=True,
)

register_template(
    name="45t-x1",
    format_user=StringFormatter(slots=["<|im_start|>user\n", "{{content}}<|im_end|>\n\n"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"token": "<|im_end|>"}]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n\n"]),
    format_function=StringFormatter(slots=["<|im_start|>assistant\n<tool_call>{{content}}</tool_call><|im_end|>\n\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n\n"]),
    format_tools=StringFormatter(slots=["<tool_list>\n{{content}}</tool_list><|im_end|>\n\n"]),
    thought_words=("<think>", "</think>"),
    replace_jinja_template=True,
)
