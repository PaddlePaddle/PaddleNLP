# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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

from typing import Dict, List, Optional

import logging
from abc import abstractmethod
from time import perf_counter
from functools import wraps
from tqdm import tqdm
from copy import deepcopy

from pipelines.schema import Document
from pipelines.nodes.base import BaseComponent
from pipelines.document_stores.base import BaseDocumentStore, BaseKnowledgeGraph

logger = logging.getLogger(__name__)


class BaseGraphRetriever(BaseComponent):
    """
    Base classfor knowledge graph retrievers.
    """

    knowledge_graph: BaseKnowledgeGraph
    outgoing_edges = 1

    @abstractmethod
    def retrieve(self, query: str, top_k: int):
        pass

    def eval(self):
        raise NotImplementedError

    def run(self, query: str, top_k: int):  # type: ignore
        answers = self.retrieve(query=query, top_k=top_k)
        results = {"answers": answers}
        return results, "output_1"


class BaseRetriever(BaseComponent):
    """
    Base class for regular retrievers.
    """

    document_store: BaseDocumentStore
    outgoing_edges = 1
    query_count = 0
    index_count = 0
    query_time = 0.0
    index_time = 0.0
    retrieve_time = 0.0

    @abstractmethod
    def retrieve(
        self,
        query: str,
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        pass

    def timing(self, fn, attr_name):
        """Wrapper method used to time functions."""

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if attr_name not in self.__dict__:
                self.__dict__[attr_name] = 0
            tic = perf_counter()
            ret = fn(*args, **kwargs)
            toc = perf_counter()
            self.__dict__[attr_name] += toc - tic
            return ret

        return wrapper

    def run(  # type: ignore
        self,
        root_node: str,
        query: Optional[str] = None,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
        documents: Optional[List[dict]] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if root_node == "Query":
            self.query_count += 1
            run_query_timed = self.timing(self.run_query, "query_time")
            output, stream = run_query_timed(query=query,
                                             filters=filters,
                                             top_k=top_k,
                                             index=index,
                                             headers=headers)
        elif root_node == "File":
            self.index_count += len(documents)  # type: ignore
            run_indexing = self.timing(self.run_indexing, "index_time")
            output, stream = run_indexing(documents=documents)
        else:
            raise Exception(f"Invalid root_node '{root_node}'.")
        return output, stream

    def run_query(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        documents = self.retrieve(query=query,
                                  filters=filters,
                                  top_k=top_k,
                                  index=index,
                                  headers=headers)
        document_ids = [doc.id for doc in documents]
        logger.debug(f"Retrieved documents with IDs: {document_ids}")
        output = {"documents": documents}

        return output, "output_1"

    def run_indexing(self, documents: List[dict]):
        if self.__class__.__name__ in [
                "DensePassageRetriever", "EmbeddingRetriever"
        ]:
            documents = deepcopy(documents)
            document_objects = [Document.from_dict(doc) for doc in documents]
            embeddings = self.embed_documents(document_objects)  # type: ignore
            for doc, emb in zip(documents, embeddings):
                doc["embedding"] = emb
        output = {"documents": documents}
        return output, "output_1"

    def print_time(self):
        print("Retriever (Speed)")
        print("---------------")
        if not self.index_count:
            print("No indexing performed via Retriever.run()")
        else:
            print(f"Documents indexed: {self.index_count}")
            print(f"Index time: {self.index_time}s")
            print(f"{self.query_time / self.query_count} seconds per document")
        if not self.query_count:
            print("No querying performed via Retriever.run()")
        else:
            print(f"Queries Performed: {self.query_count}")
            print(f"Query time: {self.query_time}s")
            print(f"{self.query_time / self.query_count} seconds per query")
