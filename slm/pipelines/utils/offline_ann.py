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

import argparse
import time

from pipelines.document_stores import (
    BaiduElasticsearchDocumentStore,
    ElasticsearchDocumentStore,
    MilvusDocumentStore,
)
from pipelines.nodes import DensePassageRetriever
from pipelines.utils import convert_files_to_dicts, fetch_archive_from_http, launch_es
from pipelines.utils.preprocessing import convert_files_to_dicts_splitter

data_dict = {
    "data/dureader_dev": "https://paddlenlp.bj.bcebos.com/applications/dureader_dev.zip",
    "data/baike": "https://paddlenlp.bj.bcebos.com/applications/baike.zip",
    "data/insurance": "https://paddlenlp.bj.bcebos.com/applications/insurance.zip",
    "data/file_example": "https://paddlenlp.bj.bcebos.com/pipelines/file_examples.zip",
}

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--index_name", default="baike_cities", type=str, help="The index name of the ANN search engine")
parser.add_argument("--doc_dir", default="data/baike/", type=str, help="The doc path of the corpus")
parser.add_argument('--username', type=str, default="", help='Username of ANN search engine')
parser.add_argument('--password', type=str, default="", help='Password of ANN search engine')
parser.add_argument("--search_engine", choices=["elastic", "milvus", 'bes'], default="elastic", help="The type of ANN search engine.")
parser.add_argument("--host", type=str, default="127.0.0.1", help="host ip of ANN search engine")
parser.add_argument("--port", type=str, default="9200", help="port of ANN search engine")
parser.add_argument("--embedding_dim", default=768, type=int, help="The embedding_dim of index")
parser.add_argument("--split_answers", action="store_true", help="whether to split lines into question and answers")
parser.add_argument("--query_embedding_model", default="rocketqa-zh-base-query-encoder", type=str, help="The query_embedding_model path",)
parser.add_argument("--passage_embedding_model", default="rocketqa-zh-base-para-encoder", type=str, help="The passage_embedding_model path", )
parser.add_argument("--params_path", default="checkpoints/model_40/model_state.pdparams", type=str, help="The checkpoint path")
parser.add_argument("--delete_index", action="store_true", help="Whether to delete existing index while updating index")
parser.add_argument("--share_parameters", action="store_true", help="Use to control the query and title models sharing the same parameters",)
parser.add_argument('--model_type', choices=['ernie_search', 'ernie', 'bert', 'neural_search'], default="ernie", help="the ernie model types")
parser.add_argument('--embed_title', default=False, type=bool, help="The title to be  embedded into embedding")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select devices, defaults to gpu.")
parser.add_argument('--search_fields', default=['content', 'name'], help="multi recall BM25Retriever set search_fields")
parser.add_argument('--use_splitter', default=False, type=bool, help="How to split documents")
parser.add_argument('--chunk_size', type=int, default=300, help="The length of data for indexing by retriever")
parser.add_argument('--chunk_overlap', type=int, default=0, help="a larger chunk than the chunk overlap")
parser.add_argument('--separator', type=str, default='\n', help="Use symbols to segment text, PDF, and image files, or connect some short chunks")
parser.add_argument('--filters', type=list, default=['\n'], help="Filter special symbols")
parser.add_argument('--language', type=str, default='chinese', help="the language of files")
parser.add_argument('--pooling_mode', choices=['max_tokens', 'mean_tokens', 'mean_sqrt_len_tokens', 'cls_token'], default='cls_token', help='the type of sentence embedding')
parser.add_argument("--es_chunk_size", default=500, type=int, help="Number of docs in one chunk sent to es")
parser.add_argument("--es_thread_count", default=32, type=int, help="Size of the threadpool to use for the bulk requests")
parser.add_argument("--es_queue_size", default=32, type=int, help="Size of the task queue between the main thread (producing chunks to send) and the processing threads.")
args = parser.parse_args()
# yapf: enable


def offline_ann(index_name, doc_dir):
    use_gpu = True if args.device == "gpu" else False
    if args.search_engine == "milvus":
        document_store = MilvusDocumentStore(
            embedding_dim=args.embedding_dim,
            host=args.host,
            index=args.index_name,
            port=args.port,
            index_param={"M": 16, "efConstruction": 50},
            index_type="HNSW",
        )
    elif args.search_engine == "bes":

        document_store = BaiduElasticsearchDocumentStore(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            embedding_dim=args.embedding_dim,
            similarity="dot_prod",
            vector_type="bpack_vector",
            search_fields=["content", "meta"],
            index=args.index_name,
            chunk_size=args.es_chunk_size,
            thread_count=args.es_thread_count,
            queue_size=args.es_queue_size,
        )

    else:
        launch_es()
        document_store = ElasticsearchDocumentStore(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            embedding_dim=args.embedding_dim,
            index=index_name,
            search_fields=args.search_fields,  # 当使用了多路召回并且搜索字段设置了除content的其他字段，构建索引时其他字段也需要设置，例如：['content', 'name']。
        )
    # 将每篇文档按照段落进行切分
    if args.use_splitter:
        dicts = convert_files_to_dicts_splitter(
            dir_path=doc_dir,
            split_paragraphs=True,
            split_answers=args.split_answers,
            encoding="utf-8",
            separator=args.separator,
            filters=args.filters,
            chunk_size=args.chunk_size,
            language=args.language,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        dicts = convert_files_to_dicts(
            dir_path=doc_dir, split_paragraphs=True, split_answers=args.split_answers, encoding="utf-8"
        )

    print(dicts[:3])

    # 文档数据写入数据库
    document_store.write_documents(dicts)
    # 语义索引模型
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=args.query_embedding_model,
        passage_embedding_model=args.passage_embedding_model,
        params_path=args.params_path,
        output_emb_size=args.embedding_dim if args.model_type in ["ernie_search", "neural_search"] else None,
        share_parameters=args.share_parameters,
        max_seq_len_query=64,
        max_seq_len_passage=256,
        batch_size=16,
        use_gpu=use_gpu,
        embed_title=args.embed_title,
    )
    # Writing docs may take a while. so waitting until writing docs to be completed.
    document_count = document_store.get_document_count()
    while document_count == 0:
        time.sleep(1)
        print("Waiting for writing docs to be completed.")
        document_count = document_store.get_document_count()
    # 建立索引库
    document_store.update_embeddings(retriever)


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
    elif args.search_engine == "bes":

        document_store = BaiduElasticsearchDocumentStore(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            embedding_dim=args.embedding_dim,
            similarity="dot_prod",
            vector_type="bpack_vector",
            search_fields=["content", "meta"],
            index=args.index_name,
            chunk_size=args.es_chunk_size,
            thread_count=args.es_thread_count,
            queue_size=args.es_queue_size,
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
