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
# flake8: noqa
import importlib
import os

from pipelines.document_stores.base import (
    BaseDocumentStore,
    BaseKnowledgeGraph,
    KeywordDocumentStore,
)
from pipelines.utils.import_utils import safe_import

ElasticsearchDocumentStore = safe_import(
    "pipelines.document_stores.elasticsearch", "ElasticsearchDocumentStore", "elasticsearch"
)
OpenDistroElasticsearchDocumentStore = safe_import(
    "pipelines.document_stores.elasticsearch", "OpenDistroElasticsearchDocumentStore", "elasticsearch"
)
OpenSearchDocumentStore = safe_import(
    "pipelines.document_stores.elasticsearch", "OpenSearchDocumentStore", "elasticsearch"
)

BaiduElasticsearchDocumentStore = safe_import(
    "pipelines.document_stores.elasticsearch", "BaiduElasticsearchDocumentStore", "elasticsearch"
)

FAISSDocumentStore = safe_import("pipelines.document_stores.faiss", "FAISSDocumentStore", "faiss")

MilvusDocumentStore = safe_import("pipelines.document_stores.milvus2", "Milvus2DocumentStore", "milvus")

from pipelines.document_stores.utils import (
    es_index_to_document_store,
    eval_data_from_json,
    eval_data_from_jsonl,
)
