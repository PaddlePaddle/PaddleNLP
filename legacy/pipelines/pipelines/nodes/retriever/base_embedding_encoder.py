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
from abc import abstractmethod
from typing import List

import numpy as np
from pipelines.document_stores import BaseDocumentStore
from pipelines.schema import Document

logger = logging.getLogger(__name__)


class _BaseEmbeddingEncoder:
    @abstractmethod
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        pass

    @abstractmethod
    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param docs: List of documents to embed.
        :return: Embeddings, one per input document, shape: (documents, embedding_dim)
        """
        pass

    def _check_docstore_similarity_function(self, document_store: BaseDocumentStore, model_name: str):
        """
        Check that document_store uses a similarity function
        compatible with the embedding model
        """
        if "dpr" in model_name.lower() and document_store.similarity != "dot_product":
            logger.warning(
                "You seem to be using a DPR model with the %s function. "
                "We recommend using dot_product instead. "
                "This can be set when initializing the DocumentStore",
                document_store.similarity,
            )
