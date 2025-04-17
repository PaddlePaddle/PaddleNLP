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

import logging
from typing import Any, List, Optional, Tuple

from pipelines.nodes import ErnieBot
from pipelines.nodes.combine_documents.base import BaseCombineDocuments

logger = logging.getLogger(__name__)


def format_document(doc: dict, prompt: str) -> str:
    return prompt.format(**doc)


class StuffDocuments(BaseCombineDocuments):
    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        llm=None,
        document_prompt: str = "文件{index}: 文件内容{content}",
        document_separator: str = "\n\n",
        llm_prompt: Optional[str] = None,
        len_str: int = 10000,
        **kwargs
    ):
        super().__init__(**kwargs)

        """
        The StuffDocuments class is a subclass of the BaseCombineDocuments class,
        which is designed to generate multi document summary.
        First, merge multiple documents, and then generate a multi document summary .
        Ensuring that the number of tokens for all documents does not exceed the len_str.

        :param document_prompt: the prompt for geting and merging multiple documents
        :param llm_prompt: the prompt for multiple document summaries
        :param len_str: maximum document length
        :param llm: the  Language Model
        """
        self.document_prompt = document_prompt
        self.document_separator = document_separator
        self.llm_prompt = llm_prompt
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ErnieBot(api_key, secret_key)
        self.len_str = len_str
        """
        示例：
        document_prompt='文件{index}: 文件内容{content}'
        llm_prompt='根据下列多个的文件内容给出一个摘要：\n{}'
        """

    def _get_inputs(self, docs: List[dict], **kwargs: Any) -> str:
        # Format each document according to the prompt
        for index, doc in enumerate(docs):
            docs[index]["index"] = index
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
        # Join the documents together to put them in the prompt.
        return self.document_separator.join(doc_strings)

    def get_num_tokens(self, prompt):
        return len(prompt)

    def prompt_length(self, docs: List[dict], **kwargs: Any) -> Optional[int]:
        inputs = self._get_inputs(docs, **kwargs)
        prompt = self.llm_prompt.format(inputs)
        return self.get_num_tokens(prompt)

    def combine_docs(self, docs: List[dict], **kwargs: Any) -> Tuple[dict, str]:
        # Merge multiple files into one file
        inputs = self._get_inputs(docs, **kwargs)
        if self.llm_prompt is not None:
            inputs = self.llm_prompt.format(inputs)
        if len(inputs) > self.len_str:
            logger.info("the length of text is too long")
            inputs = inputs[: self.len_str]
        # Call predict on the LLM.
        return self.llm.run(inputs)
