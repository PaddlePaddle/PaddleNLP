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
from pipelines.nodes.combine_documents.reduce import ReduceDocuments

logger = logging.getLogger(__name__)


class MapReduceDocuments(BaseCombineDocuments):
    def __init__(
        self,
        llm_prompt: str,
        reduce_documents: BaseCombineDocuments,
        api_key: str = "",
        secret_key: str = "",
        llm=None,
        **kwargs,
    ):
        """
        The MapReduceDocuments class is a subclass of the BaseCombineDocuments class,
        which is designed to implement multiple document summaries.
        It first conducts a single document summary, followed by a collapsed multi document summary.

        :param llm_prompt: the prompt for single document summary
        :param reduce_documents: the collapse multi document summary generation
        :param token_max: the maximum length of collapsing documents
        :param llm: the  Language Model
        """
        self.llm_prompt = llm_prompt
        self.reduce_documents = reduce_documents
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ErnieBot(api_key, secret_key)
        assert isinstance(
            self.reduce_documents, ReduceDocuments
        ), f"`reduce_documents` is of type {type(self.reduce_documents)} so it does not have this attribute."

    def combine_docs(
        self,
        docs: List[dict],
        token_max: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        map_results = []
        for doc in docs:
            result = self.llm.run(self.llm_prompt.format(doc["content"]))[0]
            try:
                txt = result["result"]
            except KeyError:
                logger.info("The text parsing error")
                txt = doc["content"]
            map_results.append(txt)
        result_docs = []
        for i, r in enumerate(map_results):
            if "meta" in docs[i]:
                result_docs.append({"content": r, "meta": docs[i]["meta"]})
            else:
                result_docs.append({"content": r, "meta": {}})
        result = self.reduce_documents.combine_docs(result_docs, token_max=token_max, **kwargs)
        return result
