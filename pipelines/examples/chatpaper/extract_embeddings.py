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
import os
from concurrent.futures import ThreadPoolExecutor

import jsonlines
import pandas as pd

from pipelines.document_stores import (
    BaiduElasticsearchDocumentStore,
    FAISSDocumentStore,
)
from pipelines.nodes.retriever.parallel_retriever import ParallelRetriever

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
parser.add_argument('--start_idx', default=0, type=int, help='The file count start index')
parser.add_argument('--end_idx', default=-1, type=int, help='The file count end index')
parser.add_argument("--search_engine", choices=['faiss', 'bes'], default="faiss", help="The type of ANN search engine.")
parser.add_argument("--es_chunk_size", default=500, type=int, help="Number of docs in one chunk sent to es")
args = parser.parse_args()
# yapf: enable


def read_data(file_path):
    data = pd.read_json(path_or_buf=file_path, lines=True)
    list_data = []
    for index, row in data.iterrows():
        doc = row.to_dict()
        list_data.append(doc)
    return list_data


def embedding_extraction(file_path):
    print(f"processing {file_path}")
    docs = read_data(file_path)
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
        model_name="bge",
    )
    log_file = open("log_embeddings.txt", "a")
    try:
        # Manually indexing
        res = retriever.run_indexing(docs)
        documents = res[0]["documents"]
        print(len(docs))
        log_file.write(file_path + "\t" + "success" + "\n")

    except Exception as e:
        print("Indexing failed, please try again.")
        log_file.write(file_path + "\t" + e + "\n")

    file_name = file_path.split("/")[-1]
    file_name = file_name.split(".")[0]
    json_name = f"{root_path}/{file_name}_embeddings.jsonl"
    # json_name = f'data/{file_name}_embeddings.jsonl'
    with jsonlines.open(json_name, mode="w") as writer:
        for doc in documents:
            # print(doc["embedding"].shape)
            doc["embedding"] = doc["embedding"].tolist()
            writer.write(doc)


if __name__ == "__main__":

    file_paths = glob.glob("*.jsonl")
    file_paths.sort()
    root_path = os.path.join("data", "full_text_embeddings")
    os.makedirs(root_path, exist_ok=True)
    file_paths = file_paths[args.start_idx : args.end_idx]
    thread_count = 2
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        executor.map(embedding_extraction, file_paths)
