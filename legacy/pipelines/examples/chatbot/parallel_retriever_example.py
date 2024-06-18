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

from pipelines.document_stores import (
    BaiduElasticsearchDocumentStore,
    FAISSDocumentStore,
)
from pipelines.nodes import (
    ErnieBot,
    ErnieRanker,
    PromptTemplate,
    TruncatedConversationHistory,
)
from pipelines.nodes.file_converter import TextConverter
from pipelines.nodes.preprocessor.text_splitter import CharacterTextSplitter
from pipelines.nodes.retriever.parallel_retriever import ParallelRetriever
from pipelines.pipelines import Pipeline

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to run dense_qa system, defaults to gpu.")
parser.add_argument("--file_paths", default='./data/md_files', type=str, help="The PDF file path.")
parser.add_argument("--max_seq_len_query", default=64, type=int, help="The maximum total length of query after tokenization.")
parser.add_argument("--max_seq_len_passage", default=256, type=int, help="The maximum total length of passage after tokenization.")
parser.add_argument("--retriever_batch_size", default=16, type=int, help="The batch size of retriever to extract passage embedding for building ANN index.")
parser.add_argument("--params_path", default="checkpoints/model_40/model_state.pdparams", type=str, help="The checkpoint path")
parser.add_argument("--data_chunk_size", default=300, type=int, help="The length of data for indexing by retriever")
parser.add_argument('--host', type=str, default="localhost", help='host ip of ANN search engine')
parser.add_argument('--embed_title', default=False, type=bool, help="The title to be  embedded into embedding")
parser.add_argument('--model_type', choices=['ernie_search', 'ernie', 'bert', 'neural_search'], default="ernie", help="the ernie model types")
parser.add_argument('--title_split', default=False, type=bool, help='the markdown file is split by titles')
parser.add_argument("--api_key", default=None, type=str, help="The API Key.")
parser.add_argument("--secret_key", default=None, type=str, help="The secret key.")
parser.add_argument('--url', default='0.0.0.0:8082', type=str, help='The port of the HTTP service')
parser.add_argument('--num_process', default=10, type=int, help='The number of process used for parallel retriever')
parser.add_argument("--port", type=str, default="9200", help="port of ANN search engine")
parser.add_argument("--es_thread_count", default=32, type=int, help="Size of the threadpool to use for the bulk requests")
parser.add_argument("--es_queue_size", default=32, type=int, help="Size of the task queue between the main thread (producing chunks to send) and the processing threads.")
parser.add_argument("--index_name", default='dureader_index', type=str, help="The ann index name of ANN.")
parser.add_argument('--username', type=str, default="", help='Username of ANN search engine')
parser.add_argument('--password', type=str, default="", help='Password of ANN search engine')
parser.add_argument("--search_engine", choices=['faiss', 'bes'], default="faiss", help="The type of ANN search engine.")
parser.add_argument("--es_chunk_size", default=500, type=int, help="Number of docs in one chunk sent to es")
args = parser.parse_args()
# yapf: enable


def ChatFile():
    txt_converter = TextConverter()
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=args.data_chunk_size, chunk_overlap=0, filters=["\n"]
    )
    if args.search_engine == "faiss":
        document_store = FAISSDocumentStore(embedding_dim=768, faiss_index_factory_str="Flat", return_embedding=True)
    else:
        document_store = BaiduElasticsearchDocumentStore(
            embedding_dim=768,
            duplicate_documents="skip",
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            index=args.index_name,
            similarity="dot_prod",
            vector_type="bpack_vector",
            search_fields=["content", "meta"],
            chunk_size=args.es_chunk_size,
            thread_count=args.es_thread_count,
            queue_size=args.es_queue_size,
        )
    use_gpu = True if args.device == "gpu" else False
    retriever = ParallelRetriever(
        document_store=document_store,
        params_path=args.params_path,
        output_emb_size=None,
        max_seq_len_query=args.max_seq_len_query,
        max_seq_len_passage=args.max_seq_len_passage,
        use_gpu=use_gpu,
        batch_size=args.retriever_batch_size,
        embed_title=args.embed_title,
        url=args.url,
        num_process=args.num_process,
    )
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=txt_converter, name="txt_converter", inputs=["File"])
    indexing_pipeline.add_node(component=text_splitter, name="Splitter", inputs=["txt_converter"])
    indexing_pipeline.add_node(component=retriever, name="Retriever", inputs=["Splitter"])
    indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Retriever"])
    files = glob.glob(args.file_paths + "/**/*.txt", recursive=True)
    indexing_pipeline.run(file_paths=files)
    ernie_bot = ErnieBot(api_key=args.api_key, secret_key=args.secret_key)
    ranker = ErnieRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=True)
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
    query_pipeline.add_node(component=PromptTemplate("背景：{documents} 问题：{query}"), name="Template", inputs=["Ranker"])
    query_pipeline.add_node(
        component=TruncatedConversationHistory(max_length=256), name="TruncateHistory", inputs=["Template"]
    )
    query_pipeline.add_node(component=ernie_bot, name="ErnieBot", inputs=["TruncateHistory"])
    query = "高血压出现头晕，脸红的症状怎么办？"
    prediction = query_pipeline.run(query=query, params={"Retriever": {"top_k": 30}, "Ranker": {"top_k": 3}})
    print("user: {}".format(query))
    print("assistant: {}".format(prediction["result"]))


if __name__ == "__main__":
    ChatFile()
