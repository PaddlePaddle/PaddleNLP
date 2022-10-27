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
import os

import paddle
from pipelines.utils import convert_files_to_dicts, fetch_archive_from_http
from pipelines.document_stores import ElasticsearchDocumentStore, MilvusDocumentStore
from pipelines.nodes import DensePassageRetriever
from pipelines.utils import launch_es

data_dict = {
    'data/dureader_dev':
    "https://paddlenlp.bj.bcebos.com/applications/dureader_dev.zip",
    "data/baike":
    "https://paddlenlp.bj.bcebos.com/applications/baike.zip",
    "data/insurance":
    "https://paddlenlp.bj.bcebos.com/applications/insurance.zip",
    "data/file_example":
    "https://paddlenlp.bj.bcebos.com/pipelines/file_examples.zip"
}

parser = argparse.ArgumentParser()
parser.add_argument("--index_name",
                    default='baike_cities',
                    type=str,
                    help="The index name of the ANN search engine")
parser.add_argument("--doc_dir",
                    default='data/baike/',
                    type=str,
                    help="The doc path of the corpus")
parser.add_argument("--search_engine",
                    choices=['elastic', 'milvus'],
                    default="elastic",
                    help="The type of ANN search engine.")
parser.add_argument('--host',
                    type=str,
                    default="127.0.0.1",
                    help='host ip of ANN search engine')

parser.add_argument('--port',
                    type=str,
                    default="9200",
                    help='port of ANN search engine')

parser.add_argument("--embedding_dim",
                    default=312,
                    type=int,
                    help="The embedding_dim of index")

parser.add_argument('--split_answers',
                    action='store_true',
                    help='whether to split lines into question and answers')

parser.add_argument("--query_embedding_model",
                    default="rocketqa-zh-nano-query-encoder",
                    type=str,
                    help="The query_embedding_model path")

parser.add_argument("--passage_embedding_model",
                    default="rocketqa-zh-nano-para-encoder",
                    type=str,
                    help="The passage_embedding_model path")

parser.add_argument("--params_path",
                    default="checkpoints/model_40/model_state.pdparams",
                    type=str,
                    help="The checkpoint path")

parser.add_argument(
    '--delete_index',
    action='store_true',
    help='whether to delete existing index while updating index')

args = parser.parse_args()


def offline_ann(index_name, doc_dir):

    if (args.search_engine == "milvus"):
        document_store = MilvusDocumentStore(embedding_dim=args.embedding_dim,
                                             host=args.host,
                                             index=args.index_name,
                                             port=args.port,
                                             index_param={
                                                 "M": 16,
                                                 "efConstruction": 50
                                             },
                                             index_type="HNSW")
    else:
        launch_es()
        document_store = ElasticsearchDocumentStore(
            host=args.host,
            port=args.port,
            username="",
            password="",
            embedding_dim=args.embedding_dim,
            index=index_name)
    # 将每篇文档按照段落进行切分
    dicts = convert_files_to_dicts(dir_path=doc_dir,
                                   split_paragraphs=True,
                                   split_answers=args.split_answers,
                                   encoding='utf-8')

    print(dicts[:3])

    # 文档数据写入数据库
    document_store.write_documents(dicts)

    ### 语义索引模型
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=args.query_embedding_model,
        passage_embedding_model=args.passage_embedding_model,
        params_path=args.params_path,
        output_emb_size=args.embedding_dim,
        max_seq_len_query=64,
        max_seq_len_passage=256,
        batch_size=16,
        use_gpu=True,
        embed_title=False,
    )

    # 建立索引库
    document_store.update_embeddings(retriever)


def delete_data(index_name):
    if (args.search_engine == 'milvus'):
        document_store = MilvusDocumentStore(embedding_dim=args.embedding_dim,
                                             host=args.host,
                                             index=args.index_name,
                                             port=args.port,
                                             index_param={
                                                 "M": 16,
                                                 "efConstruction": 50
                                             },
                                             index_type="HNSW")
    else:
        document_store = ElasticsearchDocumentStore(
            host=args.host,
            port=args.port,
            username="",
            password="",
            embedding_dim=args.embedding_dim,
            index=index_name)
    document_store.delete_index(index_name)
    print('Delete an existing elasticsearch index {} Done.'.format(index_name))


if __name__ == "__main__":
    if (args.doc_dir in data_dict):
        fetch_archive_from_http(url=data_dict[args.doc_dir],
                                output_dir=args.doc_dir)
    if (args.delete_index):
        delete_data(args.index_name)
    offline_ann(args.index_name, args.doc_dir)
