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

from pipelines.document_stores import (
    BaiduElasticsearchDocumentStore,
    ElasticsearchDocumentStore,
)
from pipelines.nodes import (
    BM25Retriever,
    DensePassageRetriever,
    ErnieRanker,
    JoinDocuments,
)
from pipelines.pipelines import Pipeline
from pipelines.utils import (
    convert_files_to_dicts_splitter,
    fetch_archive_from_http,
    print_documents,
)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to run dense_qa system, defaults to gpu.")
parser.add_argument("--index_name", default='dureader_nano_query_encoder', type=str, help="The ann index name of ANN.")
parser.add_argument('--username', type=str, default="", help='Username of ANN search engine')
parser.add_argument('--password', type=str, default="", help='Password of ANN search engine')
parser.add_argument("--search_engine", choices=['elastic', 'bes'], default="elastic", help="The type of ANN search engine.")
parser.add_argument("--max_seq_len_query", default=64, type=int, help="The maximum total length of query after tokenization.")
parser.add_argument("--max_seq_len_passage", default=384, type=int, help="The maximum total length of passage after tokenization.")
parser.add_argument("--retriever_batch_size", default=16, type=int, help="The batch size of retriever to extract passage embedding for building ANN index.")
parser.add_argument("--query_embedding_model", default="rocketqa-zh-nano-query-encoder", type=str, help="The query_embedding_model path")
parser.add_argument("--passage_embedding_model", default="rocketqa-zh-nano-para-encoder", type=str, help="The passage_embedding_model path")
parser.add_argument("--params_path", default="", type=str, help="The checkpoint path")
parser.add_argument("--embedding_dim", default=312, type=int, help="The embedding_dim of index")
parser.add_argument('--host', type=str, default="localhost", help='host ip of ANN search engine')
parser.add_argument('--port', type=str, default="9200", help='port of ANN search engine')
parser.add_argument("--bm_topk", default=10, type=int, help="The number of candidates for BM25Retriever to retrieve.")
parser.add_argument("--dense_topk", default=10, type=int, help="The number of candidates for DensePassageRetriever to retrieve.")
parser.add_argument("--rank_topk", default=10, type=int, help="The number of candidates ranker to filter.")

args = parser.parse_args()
# yapf: enable


def get_retrievers(use_gpu):

    doc_dir = "data/dureader_dev"
    dureader_data = "https://paddlenlp.bj.bcebos.com/applications/dureader_dev.zip"
    fetch_archive_from_http(url=dureader_data, output_dir=doc_dir)
    dicts = convert_files_to_dicts_splitter(dir_path=doc_dir, split_paragraphs=True, encoding="utf-8")
    if args.search_engine == "elastic":
        document_store_with_docs = ElasticsearchDocumentStore(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            embedding_dim=args.embedding_dim,
            vector_type="dense_vector",
            search_fields=["content", "meta"],
            index=args.index_name,
        )
    else:
        document_store_with_docs = BaiduElasticsearchDocumentStore(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            embedding_dim=args.embedding_dim,
            similarity="dot_prod",
            vector_type="bpack_vector",
            search_fields=["content", "meta"],
            index=args.index_name,
        )

    document_store_with_docs.write_documents(dicts)

    dpr_retriever = DensePassageRetriever(
        document_store=document_store_with_docs,
        query_embedding_model=args.query_embedding_model,
        passage_embedding_model=args.passage_embedding_model,
        params_path=args.params_path,
        output_emb_size=args.embedding_dim,
        max_seq_len_query=args.max_seq_len_query,
        max_seq_len_passage=args.max_seq_len_passage,
        batch_size=args.retriever_batch_size,
        use_gpu=use_gpu,
        embed_title=False,
    )
    # update Embedding
    document_store_with_docs.update_embeddings(dpr_retriever)

    bm_retriever = BM25Retriever(document_store=document_store_with_docs)

    return dpr_retriever, bm_retriever


def semantic_search_tutorial():

    use_gpu = True if args.device == "gpu" else False

    dpr_retriever, bm_retriever = get_retrievers(use_gpu)

    # Ranker
    ranker = ErnieRanker(model_name_or_path="rocketqa-nano-cross-encoder", use_gpu=use_gpu)

    # Pipeline
    pipeline = Pipeline()
    pipeline.add_node(component=bm_retriever, name="BMRetriever", inputs=["Query"])
    pipeline.add_node(component=dpr_retriever, name="DenseRetriever", inputs=["Query"])
    pipeline.add_node(
        component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["BMRetriever", "DenseRetriever"]
    )
    pipeline.add_node(component=ranker, name="Ranker", inputs=["JoinResults"])
    # Keywords recall results
    prediction = pipeline.run(
        query="广播权",
        params={
            "BMRetriever": {"top_k": args.bm_topk},
            "DenseRetriever": {"top_k": args.dense_topk},
            "Ranker": {"top_k": args.rank_topk},
        },
    )
    print_documents(prediction)
    # Dense vector recall results
    prediction = pipeline.run(
        query="期货交易手续费指的是什么?",
        params={
            "BMRetriever": {"top_k": args.bm_topk},
            "DenseRetriever": {"top_k": args.dense_topk},
            "Ranker": {"top_k": args.rank_topk},
        },
    )
    print_documents(prediction)


if __name__ == "__main__":
    semantic_search_tutorial()
