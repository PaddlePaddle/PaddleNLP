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

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from abc import abstractmethod

import numpy as np
import paddle
from pipelines.document_stores import BaseDocumentStore
from pipelines.nodes.models import SemanticIndexBatchNeg
from pipelines.nodes.retriever.base import BaseRetriever
from pipelines.nodes.retriever.ernie_encoder import ErnieEmbeddingEncoder
from pipelines.schema import ContentTypes, Document, FilterType
from pipelines.utils.common_utils import initialize_device_settings
from tqdm.auto import tqdm

from paddlenlp import Taskflow
from paddlenlp.transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class DensePassageRetriever(BaseRetriever):
    """
    Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_embedding_model: Union[Path, str] = "rocketqa-zh-dureader-query-encoder",
        passage_embedding_model: Union[Path, str] = "rocketqa-zh-dureader-para-encoder",
        params_path: Optional[str] = "",
        model_version: Optional[str] = None,
        output_emb_size: Optional[int] = None,
        reinitialize: bool = False,
        share_parameters: bool = False,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 384,
        top_k: int = 10,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_title: bool = True,
        similarity_function: str = "dot_product",
        progress_bar: bool = True,
        mode: Literal["snippets", "raw_documents", "preprocessed_documents"] = "preprocessed_documents",
        **kwargs
    ):
        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.

        **Example:**

                ```python
                |    # remote model from FAIR
                |    DensePassageRetriever(document_store=your_doc_store,
                |                          query_embedding_model="rocketqa-zh-dureader-query-encoder",
                |                          passage_embedding_model="rocketqa-zh-dureader-para-encoder")
                |    # or from local path
                |    DensePassageRetriever(document_store=your_doc_store,
                |                          query_embedding_model="model_directory/question-encoder",
                |                          passage_embedding_model="model_directory/context-encoder")
                ```
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by paddlenlp transformers' models
                                      Currently available remote names: ``"rocketqa-zh-dureader-query-encoder"``
        :param passage_embedding_model: Local path or remote name of passage encoder checkpoint. The format equals the
                                        one used by paddlenlp transformers' models
                                        Currently available remote names: ``"rocketqa-zh-dureader-para-encoder"``
        :param max_seq_len_query: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
        :param max_seq_len_passage: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
        :param top_k: How many documents to return per query.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_title: Whether to concatenate title and passage to a text pair that is then used to create the embedding.
                            This is the approach used in the original paper and is likely to improve performance if your
                            titles contain meaningful information for retrieval (topic, entities etc.) .
                            The title is expected to be present in doc.meta["name"] and can be supplied in the documents
                            before writing them to the DocumentStore like this:
                            {"text": "my text", "meta": {"name": "my title"}}.
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        """
        # Save init parameters to enable export of component config as YAML
        self.set_config(
            document_store=document_store,
            query_embedding_model=query_embedding_model,
            passage_embedding_model=passage_embedding_model,
            model_version=model_version,
            max_seq_len_query=max_seq_len_query,
            max_seq_len_passage=max_seq_len_passage,
            top_k=top_k,
            use_gpu=use_gpu,
            batch_size=batch_size,
            embed_title=embed_title,
            reinitialize=reinitialize,
            share_parameters=share_parameters,
            output_emb_size=output_emb_size,
            similarity_function=similarity_function,
            progress_bar=progress_bar,
        )

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)
        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices. All gpus will not be utilized.")

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.embed_title = embed_title
        self.mode = mode

        if document_store is None:
            logger.warning("DensePassageRetriever initialized without a document store. ")
        elif document_store.similarity != "dot_product":
            logger.warning(
                f"You are using a Dense Passage Retriever model with the {document_store.similarity} function. "
                "We recommend you use dot_product instead. "
                "This can be set when initializing the DocumentStore"
            )

        # Init & Load Encoders
        if os.path.exists(params_path):
            pretrained_model = AutoModel.from_pretrained(query_embedding_model)
            self.ernie_dual_encoder = SemanticIndexBatchNeg(pretrained_model, output_emb_size=output_emb_size)
            # Load Custom models
            logger.info("Loading Parameters from:{}".format(params_path))
            state_dict = paddle.load(params_path)
            self.ernie_dual_encoder.set_dict(state_dict)
            self.query_tokenizer = AutoTokenizer.from_pretrained(query_embedding_model)
            self.passage_tokenizer = AutoTokenizer.from_pretrained(query_embedding_model)
        else:
            self.query_encoder = Taskflow(
                "feature_extraction",
                model=query_embedding_model,
                batch_size=self.batch_size,
                return_tensors="np",
                max_len=max_seq_len_query,
                output_emb_size=output_emb_size,
                reinitialize=reinitialize,
                share_parameters=share_parameters,
                device_id=0 if use_gpu else -1,
                **kwargs,
            )
            self.passage_encoder = Taskflow(
                "feature_extraction",
                model=passage_embedding_model,
                batch_size=self.batch_size,
                return_tensors="np",
                max_len=max_seq_len_passage,
                output_emb_size=output_emb_size,
                reinitialize=reinitialize,
                share_parameters=share_parameters,
                device_id=0 if use_gpu else -1,
                **kwargs,
            )

    def retrieve(
        self,
        query: str,
        query_type: Optional[ContentTypes] = None,
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if top_k is None:
            top_k = self.top_k
        if not self.document_store:
            logger.error("Cannot perform retrieve() since DensePassageRetriever initialized with document_store=None")
            return []
        if index is None:
            index = self.document_store.index

        query_emb = self.embed_queries(texts=[query], **kwargs)
        documents = self.document_store.query_by_embedding(
            query_emb=query_emb[0], top_k=top_k, filters=filters, index=index, headers=headers, return_embedding=False
        )
        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        queries_type: Optional[ContentTypes] = None,
        filters: Optional[
            Union[
                Dict[str, Union[Dict, List, str, int, float, bool]],
                List[Dict[str, Union[Dict, List, str, int, float, bool]]],
            ]
        ] = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: bool = None,
        **kwargs,
    ) -> List[List[Document]]:
        if top_k is None:
            top_k = self.top_k
        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise Exception(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)
        if index is None:
            index = self.document_store.index
        if not self.document_store:
            logger.error(
                "Cannot perform retrieve_batch() since DensePassageRetriever initialized with document_store=None"
            )
            return [[] * len(queries)]  # type: ignore
        documents = []
        query_embs: List[np.ndarray] = []
        for batch in self._get_batches(queries=queries, batch_size=batch_size):
            query_embs.extend(self.embed_queries(texts=batch, **kwargs))
        for query_emb, cur_filters in tqdm(
            zip(query_embs, filters), total=len(query_embs), disable=not self.progress_bar, desc="Querying"
        ):
            cur_docs = self.document_store.query_by_embedding(
                query_emb=query_emb,
                top_k=top_k,
                filters=cur_filters,
                index=index,
                headers=headers,
                return_embedding=False,
            )
            documents.append(cur_docs)
        return documents

    def _get_predictions(self, dicts, **kwargs):
        """
        Feed a preprocessed dataset to the model and get the actual predictions (forward pass + formatting).

        :param dicts: list of dictionaries
        examples:[{'query': "where is florida?"}, {'query': "who wrote lord of the rings?"}, ...]
                [{'passages': [{
                    "title": 'Big Little Lies (TV series)',
                    "text": 'series garnered several accolades. It received..',
                    "label": 'positive',
                    "external_id": '18768923'},
                    {"title": 'Framlingham Castle',
                    "text": 'Castle on the Hill "Castle on the Hill" is a song by English..',
                    "label": 'positive',
                    "external_id": '19930582'}, ...]
        :return: dictionary of embeddings for "passages" and "query"
        """
        datasets = []
        if "passages" in dicts[0]:
            # dicts is a list of passages
            for passages in dicts:
                for item in passages["passages"]:
                    if self.embed_title:
                        datasets.append(item["title"] + item["text"])
                    else:
                        datasets.append(item["text"])
        elif "query" in dicts[0]:
            # dicts is a list of passages
            for passages in dicts:
                datasets.append(passages["query"])

        all_embeddings = {"query": [], "passages": []}

        # When running evaluations etc., we don't want a progress bar for every single query
        if len(datasets) == 1:
            disable_tqdm = True
        else:
            disable_tqdm = not self.progress_bar
        with tqdm(
            total=len(datasets) // self.batch_size,
            unit=" Docs",
            desc="Create embeddings",
            position=1,
            leave=False,
            disable=disable_tqdm,
        ) as progress_bar:
            for i in range(0, len(datasets), self.batch_size):

                if "query" in dicts[0]:
                    cls_embeddings = self.query_encoder(datasets[i : i + self.batch_size], **kwargs)
                    all_embeddings["query"].append(cls_embeddings["features"])
                if "passages" in dicts[0]:
                    cls_embeddings = self.passage_encoder(datasets[i : i + self.batch_size], **kwargs)
                    all_embeddings["passages"].append(cls_embeddings["features"])
                progress_bar.update(self.batch_size)

        if all_embeddings["passages"]:
            all_embeddings["passages"] = np.concatenate(all_embeddings["passages"])
        if all_embeddings["query"]:
            all_embeddings["query"] = np.concatenate(all_embeddings["query"])
        return all_embeddings

    def embed_queries(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """
        Create embeddings for a list of queries using the query encoder

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        queries = [{"query": q} for q in texts]
        result = self._get_predictions(queries, **kwargs)["query"]
        return result

    def embed_documents(self, docs: List[Document], **kwargs) -> List[np.ndarray]:
        """
        Create embeddings for a list of documents using the passage encoder

        :param docs: List of Document objects used to represent documents / passages in a standardized way within pipelines.
        :return: Embeddings of documents / passages shape (batch_size, embedding_dim)
        """
        passages = [
            {
                "passages": [
                    {
                        "title": d.meta["name"] if d.meta and "name" in d.meta else "",
                        "text": d.content,
                        "label": d.meta["label"] if d.meta and "label" in d.meta else "positive",
                        "external_id": d.id,
                    }
                ]
            }
            for d in docs
        ]
        embeddings = self._get_predictions(passages, **kwargs)["passages"]

        return embeddings


_EMBEDDING_ENCODERS = {"ernie-embedding-v1": ErnieEmbeddingEncoder}


class DenseRetriever(BaseRetriever):
    """
    Base class for all dense retrievers.
    """

    @abstractmethod
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param documents: List of documents to embed.
        :return: Embeddings of documents, one per input document, shape: (documents, embedding_dim)
        """
        pass

    def run_indexing(self, documents: List[Document]):
        documents_objects = [Document.from_dict(doc) for doc in documents]
        embeddings = self.embed_documents(documents_objects)
        for doc, emb in zip(documents, embeddings):
            doc["embedding"] = emb
        output = {"documents": documents}
        return output, "output_1"


class EmbeddingRetriever(DenseRetriever):

    """
    Retriever that uses a bi-encoder (query model for query, passage model for passage).
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        embedding_model: Union[Path, str] = "ernie-embedding-v1",
        max_seq_len: int = 384,
        top_k: int = 10,
        batch_size: int = 16,
        embed_title: bool = True,
        similarity_function: str = "dot_product",
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        scale_score: bool = True,
        progress_bar: bool = True,
        embed_meta_fields: Optional[List[str]] = None,
        mode: Literal["snippets", "raw_documents", "preprocessed_documents"] = "preprocessed_documents",
        **kwargs
    ):

        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by paddlenlp transformers' models
                                      Currently available remote names: ``"ernie-embedding-v1"``
        :param top_k: How many documents to return per query.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_title: Whether to concatenate title and passage to a text pair that is then used to create the embedding.
                            This is the approach used in the original paper and is likely to improve performance if your
                            titles contain meaningful information for retrieval (topic, entities etc.) .
                            The title is expected to be present in doc.meta["name"] and can be supplied in the documents
                            before writing them to the DocumentStore like this:
                            {"text": "my text", "meta": {"name": "my title"}}.
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        """
        if api_key is None or secret_key is None:
            raise Exception(
                "Please apply api_key and secret_key from https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu"
            )
        if embed_meta_fields is None:
            embed_meta_fields = []
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.document_store = document_store
        self.top_k = top_k
        self.embed_title = embed_title
        self.embedding_model = embedding_model
        self.max_seq_len = max_seq_len
        self.embed_meta_fields = embed_meta_fields
        self.scale_score = scale_score
        self.embedding_encoder = _EMBEDDING_ENCODERS[self.embedding_model](retriever=self)

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through the documents in a DocumentStore and return a small number of documents
        that are most relevant to the query.

        :param query: The query
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:

                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic API_KEY'} for basic authentication)
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(queries=[query])
        documents = document_store.query_by_embedding(
            query_emb=query_emb[0], filters=filters, top_k=top_k, index=index, headers=headers
        )
        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Scan through the documents in a DocumentStore and return a small number of documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one per query).

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. Can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).

                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:

                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic API_KEY'} for basic authentication)
        :param batch_size: Number of queries to embed at a time.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve_batch() method."
            )
        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = self.batch_size

        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score

        # embed_queries is already batched within by batch_size, so no need to batch the input here
        query_embs: np.ndarray = self.embed_queries(queries=queries)
        batched_query_embs: List[np.ndarray] = []
        for i in range(0, len(query_embs), batch_size):
            batched_query_embs.extend(query_embs[i : i + batch_size])
        documents = document_store.query_by_embedding_batch(
            query_embs=batched_query_embs,
            top_k=top_k,
            filters=filters,
            index=index,
            headers=headers,
            scale_score=scale_score,
        )

        return documents

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        # for backward compatibility: cast pure str input
        if isinstance(queries, str):
            queries = [queries]
        assert isinstance(queries, list), "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"
        return self.embedding_encoder.embed_queries(queries)

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param documents: List of documents to embed.
        :return: Embeddings, one per input document, shape: (docs, embedding_dim)
        """
        documents = self._preprocess_documents(documents)
        return self.embedding_encoder.embed_documents(documents)

    def _preprocess_documents(self, docs: List[Document]) -> List[Document]:
        """
        Turns table documents into text documents by representing the table in csv format.
        This allows us to use text embedding models for table retrieval.
        It also concatenates specified meta data fields with the text representations.

        :param docs: List of documents to linearize. If the document is not a table, it is returned as is.
        :return: List of documents with meta data + linearized tables or original documents if they are not tables.
        """
        linearized_docs = []
        for doc in docs:
            doc = deepcopy(doc)
            if doc.content_type == "table":
                if isinstance(doc.content, pd.DataFrame):
                    doc.content = doc.content.to_csv(index=False)
                else:
                    raise Exception("Documents of type 'table' need to have a pd.DataFrame as content field")
            # Gather all relevant metadata fields
            meta_data_fields = []
            for key in self.embed_meta_fields:
                if key in doc.meta and doc.meta[key]:
                    if isinstance(doc.meta[key], list):
                        meta_data_fields.extend([item for item in doc.meta[key]])
                    else:
                        meta_data_fields.append(doc.meta[key])
            # Convert to type string (e.g. for ints or floats)
            meta_data_fields = [str(field) for field in meta_data_fields]
            doc.content = "\n".join(meta_data_fields + [doc.content])
            linearized_docs.append(doc)
        return linearized_docs
