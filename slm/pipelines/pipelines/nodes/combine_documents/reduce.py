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
from typing import Any, Callable, List, Optional, Protocol, Tuple

from pipelines.nodes.combine_documents.base import BaseCombineDocuments

logger = logging.getLogger(__name__)


class CombineDocsProtocol(Protocol):
    """Interface for the combine_docs method."""

    def __call__(self, docs: List[dict], **kwargs: Any) -> str:
        """Interface for the combine_docs method."""


# 保证每个文档数小于设定的tokens
# ensure that the length of documents is less than the set tokens
def _split_list_of_docs(docs: List[dict], length_func: Callable, token_max: int, **kwargs: Any) -> List[List[dict]]:
    new_result_doc_list = []
    _sub_result_docs = []
    for doc in docs:
        _sub_result_docs.append(doc)
        _num_tokens = length_func(_sub_result_docs, **kwargs)
        if _num_tokens > token_max:
            if len(_sub_result_docs) == 1:
                raise ValueError("A single document was longer than the context length," " we cannot handle this.")
            new_result_doc_list.append(_sub_result_docs[:-1])
            _sub_result_docs = _sub_result_docs[-1:]
    new_result_doc_list.append(_sub_result_docs)
    return new_result_doc_list


def _collapse_docs(
    docs: List[dict],
    combine_document_func: CombineDocsProtocol,
    **kwargs: Any,
) -> dict:
    result = combine_document_func(docs, **kwargs)
    try:
        text = result[0]["result"]
    except KeyError:
        logger.info("The text parsing error")
        text = docs[0]["content"]
    combined_metadata = {k: str(v) for k, v in docs[0]["meta"].items()}
    for doc in docs[1:]:
        for k, v in doc["meta"].items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)
    return {"content": text, "meta": combined_metadata}


class ReduceDocuments(BaseCombineDocuments):
    def __init__(
        self,
        combine_documents: BaseCombineDocuments,
        collapse_documents: Optional[BaseCombineDocuments] = None,
        token_max: int = 10000,
    ):
        """
        The ReduceDocuments class is a subclass of the BaseCombineDocuments class,
        which is designed to collapse multi document summary.
        Fisrt,the number of tokens for multiple output documents is greater than the token_max.
        Then, the ReduceDocuments collapses multiple documents  (the number of tokens for collapsing documents should not exceed the token_max).
        Ultimately, it ensures that the total number of tokens for all documents does not exceed the token_max and implements multi document summary generation.

        :param combine_documents: Generate multiple document summaries
        :param collapse_documents: Iteratively collapse multiple documents to ensure that the total number of tokens for the final merged documents is less than the set value
        :param token_max: Maximum length of collapsing documents
        """
        self.combine_documents = combine_documents
        self.collapse_documents = collapse_documents
        self.token_max = token_max

    @property
    def _collapse_node(self) -> BaseCombineDocuments:
        if self.collapse_documents is not None:
            return self.collapse_documents
        else:
            return self.combine_documents

    def combine_docs(
        self,
        docs: List[dict],
        token_max: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[str, dict]:
        result_docs, _ = self._collapse(docs, token_max=token_max, **kwargs)
        return self.combine_documents.combine_docs(docs=result_docs, **kwargs)

    def _collapse(
        self,
        docs: List[dict],
        token_max: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[List[dict], dict]:
        length_func = self.combine_documents.prompt_length
        num_tokens = length_func(docs, **kwargs)

        def _collapse_docs_func(docs: List[dict], **kwargs: Any) -> str:
            return self._collapse_node.run(documents=docs, **kwargs)

        _token_max = token_max or self.token_max
        while num_tokens is not None and num_tokens > _token_max:
            new_result_doc_list = _split_list_of_docs(docs, length_func, _token_max, **kwargs)
            docs = []
            for docs_item in new_result_doc_list:
                new_doc = _collapse_docs(docs_item, _collapse_docs_func, **kwargs)
                docs.append(new_doc)
            num_tokens = length_func(docs, **kwargs)
        return docs, {}
