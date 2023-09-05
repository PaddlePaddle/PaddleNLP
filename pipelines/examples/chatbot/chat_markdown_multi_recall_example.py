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
import glob
import time

from pipelines.document_stores import (
    BaiduElasticsearchDocumentStore,
    ElasticsearchDocumentStore,
)
from pipelines.nodes import (
    BM25Retriever,
    CharacterTextSplitter,
    ChatGLMBot,
    DensePassageRetriever,
    EmbeddingRetriever,
    ErnieBot,
    ErnieRanker,
    JoinDocuments,
    MarkdownConverter,
    PromptTemplate,
    TruncatedConversationHistory,
)
from pipelines.pipelines import Pipeline

BOT_CLASSES = {
    "chatglm": ChatGLMBot,
}

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to run dense_qa system, defaults to gpu.")
parser.add_argument("--index_name", default='dureader_index', type=str, help="The ann index name of ANN.")
parser.add_argument('--username', type=str, default="", help='Username of ANN search engine')
parser.add_argument('--password', type=str, default="", help='Password of ANN search engine')
parser.add_argument("--file_paths", default='./data/md_files', type=str, help="The PDF file path.")
parser.add_argument("--search_engine", choices=['elastic', 'bes'], default="elastic", help="The type of ANN search engine.")
parser.add_argument("--max_seq_len_query", default=64, type=int, help="The maximum total length of query after tokenization.")
parser.add_argument("--max_seq_len_passage", default=256, type=int, help="The maximum total length of passage after tokenization.")
parser.add_argument("--retriever_batch_size", default=16, type=int, help="The batch size of retriever to extract passage embedding for building ANN index.")
parser.add_argument("--query_embedding_model", default="rocketqa-zh-nano-query-encoder", type=str, help="The query_embedding_model path")
parser.add_argument("--passage_embedding_model", default="rocketqa-zh-nano-query-encoder", type=str, help="The passage_embedding_model path")
parser.add_argument("--params_path", default="checkpoints/model_40/model_state.pdparams", type=str, help="The checkpoint path")
parser.add_argument("--embedding_dim", default=312, type=int, help="The embedding_dim of index")
parser.add_argument("--data_chunk_size", default=300, type=int, help="The length of data for indexing by retriever")
parser.add_argument('--host', type=str, default="localhost", help='host ip of ANN search engine')
parser.add_argument('--embed_title', default=False, type=bool, help="The title to be  embedded into embedding")
parser.add_argument('--chatbot', choices=['ernie_bot', 'chatglm'], default="chatglm", help="The chatbot models ")
parser.add_argument('--model_type', choices=['ernie_search', 'ernie', 'bert', 'neural_search', 'ernie-embedding-v1'], default="ernie", help="the ernie model types")
parser.add_argument("--api_key", default=None, type=str, help="The API Key.")
parser.add_argument("--secret_key", default=None, type=str, help="The secret key.")
parser.add_argument("--embedding_api_key", default=None, type=str, help="The Embedding API Key.")
parser.add_argument("--embedding_secret_key", default=None, type=str, help="The Embedding secret key.")
parser.add_argument("--port", type=str, default="9200", help="port of ANN search engine")
parser.add_argument("--es_chunk_size", default=500, type=int, help="Number of docs in one chunk sent to es")
parser.add_argument("--es_thread_count", default=32, type=int, help="Size of the threadpool to use for the bulk requests")
parser.add_argument("--es_queue_size", default=32, type=int, help="Size of the task queue between the main thread (producing chunks to send) and the processing threads.")
parser.add_argument('--indexing', default=False, type=bool, help='Whether indexing is enabled.')
args = parser.parse_args()
# yapf: enable


def chat_markdown_tutorial():

    if args.search_engine == "elastic":
        document_store = ElasticsearchDocumentStore(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            embedding_dim=args.embedding_dim,
            index=args.index_name,
            chunk_size=args.es_chunk_size,
            thread_count=args.es_thread_count,
            queue_size=args.es_queue_size,
        )

    else:
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
    use_gpu = True if args.device == "gpu" else False

    if args.model_type == "ernie-embedding-v1":
        retriever = EmbeddingRetriever(
            document_store=document_store,
            retriever_batch_size=args.retriever_batch_size,
            api_key=args.embedding_api_key,
            embed_title=args.embed_title,
            secret_key=args.embedding_secret_key,
        )
    else:
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=args.query_embedding_model,
            passage_embedding_model=args.passage_embedding_model,
            params_path=args.params_path,
            output_emb_size=args.embedding_dim if args.model_type in ["ernie_search", "neural_search"] else None,
            max_seq_len_query=args.max_seq_len_query,
            max_seq_len_passage=args.max_seq_len_passage,
            batch_size=args.retriever_batch_size,
            use_gpu=use_gpu,
            embed_title=args.embed_title,
            precision="fp16",
        )
    bm_retriever = BM25Retriever(document_store=document_store)

    # Indexing Markdowns
    markdown_converter = MarkdownConverter()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=args.data_chunk_size, chunk_overlap=0, filters=["\n"]
    )
    if args.indexing:
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_node(component=markdown_converter, name="MarkdownConverter", inputs=["File"])
        indexing_pipeline.add_node(component=text_splitter, name="Splitter", inputs=["MarkdownConverter"])
        indexing_pipeline.add_node(component=retriever, name="Retriever", inputs=["Splitter"])
        indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Retriever"])
        files = glob.glob(args.file_paths + "/**/*.md", recursive=True)
        if len(files) == 0:
            raise Exception("file should not be empty")
        indexing_pipeline.run(file_paths=files)

    # Query Markdowns
    if args.chatbot in ["ernie_bot"]:
        ernie_bot = ErnieBot(api_key=args.api_key, secret_key=args.secret_key)
    else:
        ernie_bot = BOT_CLASSES[args.chatbot]()
    ranker = ErnieRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=use_gpu)
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="DenseRetriever", inputs=["Query"])
    query_pipeline.add_node(component=bm_retriever, name="BMRetriever", inputs=["Query"])
    query_pipeline.add_node(
        component=JoinDocuments(join_mode="reciprocal_rank_fusion"),
        name="JoinResults",
        inputs=["BMRetriever", "DenseRetriever"],
    )
    query_pipeline.add_node(component=ranker, name="Ranker", inputs=["JoinResults"])
    query_pipeline.add_node(component=PromptTemplate("背景：{documents} 问题：{query}"), name="Template", inputs=["Ranker"])
    query_pipeline.add_node(
        component=TruncatedConversationHistory(max_length=256), name="TruncateHistory", inputs=["Template"]
    )
    query_pipeline.add_node(component=ernie_bot, name="ErnieBot", inputs=["TruncateHistory"])
    query = "理财产品的认购期是多久？"
    start_time = time.time()
    prediction = query_pipeline.run(query=query, params={"DenseRetriever": {"top_k": 10}, "Ranker": {"top_k": 5}})
    end_time = time.time()
    print("Time cost for query markdown conversion:", end_time - start_time)
    print("user: {}".format(query))
    print("assistant: {}".format(prediction["result"]))


if __name__ == "__main__":
    chat_markdown_tutorial()
