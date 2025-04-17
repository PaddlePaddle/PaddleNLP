# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Dict, List, Optional, Union

from pipelines.document_stores import BaseDocumentStore, KeywordDocumentStore
from pipelines.nodes.retriever.base import BaseRetriever
from pipelines.schema import ContentTypes, Document


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        document_store: Optional[KeywordDocumentStore] = None,
        top_k: int = 10,
        all_terms_must_match: bool = False,
        custom_query: Optional[str] = None,
    ):
        """
        :param document_store: an instance of one of the following DocumentStores to retrieve from: ElasticsearchDocumentStore, OpenSearchDocumentStore and OpenDistroElasticsearchDocumentStore.
            If None, a document store must be passed to the retrieve method for this Retriever to work.
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
                                     Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
                                     Defaults to False.
        :param custom_query: query string as per Elasticsearch DSL with a mandatory query placeholder(query).
                             Optionally, ES `filter` clause can be added where the values of `terms` are placeholders
                             that get substituted during runtime. The placeholder(${filter_name_1}, ${filter_name_2}..)
                             names must match with the filters dict supplied in self.retrieve().
                             ::
                                 **An example custom_query:**
                                 ```python
                                |    {
                                |        "size": 10,
                                |        "query": {
                                |            "bool": {
                                |                "should": [{"multi_match": {
                                |                    "query": ${query},                 // mandatory query placeholder
                                |                    "type": "most_fields",
                                |                    "fields": ["content", "title"]}}],
                                |                "filter": [                                 // optional custom filters
                                |                    {"terms": {"year": ${years}}},
                                |                    {"terms": {"quarter": ${quarters}}},
                                |                    {"range": {"date": {"gte": ${date}}}}
                                |                    ],
                                |            }
                                |        },
                                |    }
                                 ```
                             **For this custom_query, a sample retrieve() could be:**
                             ```python
                            |    self.retrieve(query="Why did the revenue increase?",
                            |                  filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})
                            ```
                             Optionally, highlighting can be defined by specifying Elasticsearch's highlight settings.
                             See https://www.elastic.co/guide/en/elasticsearch/reference/current/highlighting.html.
                             You will find the highlighted output in the returned Document's meta field by key "highlighted".
                             ::
                                 **Example custom_query with highlighting:**
                                 ```python
                                |    {
                                |        "size": 10,
                                |        "query": {
                                |            "bool": {
                                |                "should": [{"multi_match": {
                                |                    "query": ${query},                 // mandatory query placeholder
                                |                    "type": "most_fields",
                                |                    "fields": ["content", "title"]}}],
                                |            }
                                |        },
                                |        "highlight": {             // enable highlighting
                                |            "fields": {            // for fields content and title
                                |                "content": {},
                                |                "title": {}
                                |            }
                                |        },
                                |    }
                                 ```
                                 **For this custom_query, highlighting info can be accessed by:**
                                ```python
                                |    docs = self.retrieve(query="Why did the revenue increase?")
                                |    highlighted_content = docs[0].meta["highlighted"]["content"]
                                |    highlighted_title = docs[0].meta["highlighted"]["title"]
                                ```
        :param top_k: How many documents to return per query.
        """
        super().__init__()
        self.document_store: Optional[KeywordDocumentStore] = document_store
        self.top_k = top_k
        self.custom_query = custom_query
        self.all_terms_must_match = all_terms_must_match

    def retrieve(
        self,
        query: str,
        query_type: ContentTypes = "text",
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        document_store: Optional[BaseDocumentStore] = None,
        **kwargs
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
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
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )
        if not isinstance(document_store, KeywordDocumentStore):
            raise ValueError("document_store must be a subclass of KeywordDocumentStore.")

        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = document_store.index

        documents = document_store.query(
            query=query,
            filters=filters,
            top_k=top_k,
            all_terms_must_match=self.all_terms_must_match,
            custom_query=self.custom_query,
            index=index,
            headers=headers,
        )
        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        queries_type: ContentTypes = "text",
        filters: Optional[
            Union[
                Dict[str, Union[Dict, List, str, int, float, bool]],
                List[Dict[str, Union[Dict, List, str, int, float, bool]]],
            ]
        ] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the supplied queries.
        Returns a list of lists of Documents (one per query).
        :param queries: List of query strings.
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
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :param batch_size: Not applicable.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve_batch() method."
            )
        if not isinstance(document_store, KeywordDocumentStore):
            raise ValueError("document_store must be a subclass of KeywordDocumentStore.")

        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = document_store.index

        documents = document_store.query_batch(
            queries=queries,
            filters=filters,
            top_k=top_k,
            all_terms_must_match=self.all_terms_must_match,
            custom_query=self.custom_query,
            index=index,
            headers=headers,
        )
        return documents
