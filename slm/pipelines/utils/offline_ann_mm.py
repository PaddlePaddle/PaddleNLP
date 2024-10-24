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

import argparse
import os
import time

from pipelines.document_stores import ElasticsearchDocumentStore, MilvusDocumentStore
from pipelines.nodes import MultiModalRetriever
from pipelines.schema import Document
from pipelines.utils import convert_files_to_dicts, fetch_archive_from_http, launch_es

data_dict = {
    "data/wukong_test": "https://paddlenlp.bj.bcebos.com/applications/wukong_test_demo.zip",
    "data/wukong_text": "https://paddlenlp.bj.bcebos.com/applications/wukong_text.zip",
}

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--index_name", default="wukong_test", type=str, help="The index name of the ANN search engine")
parser.add_argument("--doc_dir", default="data/wukong_test", type=str, help="The doc path of the corpus")
parser.add_argument("--search_engine", choices=["elastic", "milvus"], default="elastic", help="The type of ANN search engine.")
parser.add_argument("--host", type=str, default="127.0.0.1", help="host ip of ANN search engine")
parser.add_argument("--port", type=str, default="9200", help="port of ANN search engine")
parser.add_argument("--embedding_dim", default=768, type=int, help="The embedding_dim of index")
parser.add_argument("--embedding_type", choices=["text", "image"], default="image", help="The type of raw data for embedding.")
parser.add_argument("--query_embedding_model", default="PaddlePaddle/ernie_vil-2.0-base-zh", type=str, help="The query_embedding_model path")
parser.add_argument("--document_embedding_model", default="PaddlePaddle/ernie_vil-2.0-base-zh", type=str, help="The document_embedding_model path")
parser.add_argument("--delete_index", action="store_true", help="Whether to delete existing index while updating index")
args = parser.parse_args()
# yapf: enable


def offline_ann(index_name, doc_dir):

    if args.search_engine == "milvus":
        document_store = MilvusDocumentStore(
            embedding_dim=args.embedding_dim,
            host=args.host,
            index=args.index_name,
            port=args.port,
            index_param={"M": 16, "efConstruction": 50},
            index_type="HNSW",
        )
    else:
        launch_es()
        document_store = ElasticsearchDocumentStore(
            host=args.host,
            port=args.port,
            username="",
            password="",
            embedding_dim=args.embedding_dim,
            index=index_name,
        )
    if args.embedding_type == "image":
        docs = [
            Document(content=f"./{args.doc_dir}/{filename}", content_type="image")
            for filename in os.listdir(args.doc_dir)
        ]
    elif args.embedding_type == "text":
        docs = convert_files_to_dicts(dir_path=args.doc_dir, split_paragraphs=True, encoding="utf-8")
    else:
        raise NotImplementedError

    print(docs[:3])

    # 文档数据写入数据库
    document_store.write_documents(docs)

    if args.embedding_type == "image":
        # 文搜图，对image做embedding
        retriever_mm = MultiModalRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            query_type="text",
            document_embedding_models={"image": args.document_embedding_model},
        )
    else:
        # 图搜文，对text做embedding
        retriever_mm = MultiModalRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            query_type="image",
            document_embedding_models={"text": args.document_embedding_model},
        )
    # Writing docs may take a while. so waitting until writing docs to be completed.
    document_count = document_store.get_document_count()
    while document_count == 0:
        time.sleep(1)
        print("Waiting for writing docs to be completed.")
        document_count = document_store.get_document_count()
    # 建立索引库
    document_store.update_embeddings(retriever_mm)


def delete_data(index_name):
    if args.search_engine == "milvus":
        document_store = MilvusDocumentStore(
            embedding_dim=args.embedding_dim,
            host=args.host,
            index=args.index_name,
            port=args.port,
            index_param={"M": 16, "efConstruction": 50},
            index_type="HNSW",
        )
    else:
        document_store = ElasticsearchDocumentStore(
            host=args.host,
            port=args.port,
            username="",
            password="",
            embedding_dim=args.embedding_dim,
            index=index_name,
        )
    document_store.delete_index(index_name)
    print("Delete an existing elasticsearch index {} Done.".format(index_name))


if __name__ == "__main__":
    if args.doc_dir in data_dict:
        fetch_archive_from_http(url=data_dict[args.doc_dir], output_dir=args.doc_dir)
    if args.delete_index:
        delete_data(args.index_name)
    offline_ann(args.index_name, args.doc_dir)
