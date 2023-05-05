# Copyright 2023 The LangChain Authors. All rights reserved.
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

import copy
import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from pipelines.nodes.base import BaseComponent

logger = logging.getLogger(__name__)


class TextSplitter(BaseComponent):
    """Interface for splitting text into chunks."""

    outgoing_edges = 1

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ):
        """Create a new TextSplitter."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size " f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._filter = None

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def create_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[dict]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                new_doc = {"content": chunk, "meta": copy.deepcopy(_metadatas[i])}
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: List[dict]) -> List[dict]:
        """Split documents."""
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["meta"] for doc in documents]
        return self.create_documents(texts, metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, " f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def clean(self, documents: List[dict]):
        for special_character in self._filter:
            for doc in documents:
                doc["content"] = doc["content"].replace(special_character, "")
        return documents

    def run(  # type: ignore
        self,
        documents: Union[dict, List[dict]],
        meta: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,  # type: ignore
    ):
        ret = []
        if type(documents) == dict:  # single document
            text_splits = self.split_text(documents["content"])
            for i, txt in enumerate(text_splits):
                doc = copy.deepcopy(documents)
                doc["content"] = txt

                if "meta" not in doc.keys() or doc["meta"] is None:
                    doc["meta"] = {}

                doc["meta"]["_split_id"] = i
                ret.append(doc)

        elif type(documents) == list:  # list document
            for document in documents:
                text_splits = self.split_text(document["content"])
                for i, txt in enumerate(text_splits):
                    doc = copy.deepcopy(document)
                    doc["content"] = txt

                    if "meta" not in doc.keys() or doc["meta"] is None:
                        doc["meta"] = {}

                    doc["meta"]["_split_id"] = i
                    ret.append(doc)
        if self._filter is not None and len(self._filter) > 0:
            ret = self.clean(ret)
        result = {"documents": ret}
        return result, "output_1"


class CharacterTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters."""

    def __init__(self, separator: str = "\n\n", filters: list = [], **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator
        self._filter = filters

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        if self._separator:
            splits = text.split(self._separator)
        else:
            splits = list(text)
        return self._merge_splits(splits, self._separator)


class RecursiveCharacterTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters.
    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(self, separators: Optional[List[str]] = None, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = self._separators[-1]
        for _s in self._separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        # Now that we have the separator, split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                other_info = self.split_text(s)
                final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, separator)
            final_chunks.extend(merged_text)
        return final_chunks
