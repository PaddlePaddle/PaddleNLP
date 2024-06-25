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

import numpy as np
from config import collection_name, embedding_name, partition_tag
from milvus_util import RecallByMilvus, VecToMilvus, text_max_len
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path", default="milvus/milvus_data.csv", type=str, required=True, help="The data for vector extraction."
)
parser.add_argument(
    "--embedding_path", default="corpus_embedding.npy", type=str, required=True, help="The vector path for data."
)
parser.add_argument("--index", default=0, type=int, help="index of the vector for search")
parser.add_argument("--insert", action="store_true", help="whether to insert data")
parser.add_argument("--search", action="store_true", help="whether to search data")
parser.add_argument("--batch_size", default=100000, type=int, help="number of examples to insert each time")
args = parser.parse_args()


def read_text(file_path):
    file = open(file_path)
    id2corpus = []
    for idx, data in enumerate(file.readlines()):
        id2corpus.append(data.strip())
    return id2corpus


def milvus_data_insert(data_path, embedding_path, batch_size):
    corpus_list = read_text(data_path)
    embeddings = np.load(embedding_path)
    embedding_ids = [i for i in range(embeddings.shape[0])]
    client = VecToMilvus()
    client.drop_collection(collection_name)
    data_size = len(embedding_ids)
    for i in tqdm(range(0, data_size, batch_size)):
        cur_end = i + batch_size
        if cur_end > data_size:
            cur_end = data_size
        batch_emb = embeddings[np.arange(i, cur_end)]
        entities = [
            [j for j in range(i, cur_end, 1)],
            [corpus_list[j][: text_max_len - 1] for j in range(i, cur_end, 1)],
            batch_emb,  # field embeddings, supports numpy.ndarray and list
        ]
        client.insert(
            collection_name=collection_name, entities=entities, index_name=embedding_name, partition_tag=partition_tag
        )


def milvus_data_recall(embedding_path, index):
    embeddings = np.load(embedding_path)
    embedding_ids = [i for i in range(embeddings.shape[0])]
    recall_client = RecallByMilvus()
    if index > len(embedding_ids):
        print("Index should not be larger than embedding size")
        return
    embeddings = embeddings[np.arange(index, index + 1)]
    time_start = time.time()
    result = recall_client.search(
        embeddings, embedding_name, collection_name, partition_names=[partition_tag], output_fields=["pk", "text"]
    )
    time_end = time.time()
    sum_t = time_end - time_start
    print("time cost", sum_t, "s")
    for hits in result:
        for hit in hits:
            print(f"hit: {hit}, text field: {hit.entity.get('text')}")


if __name__ == "__main__":
    if args.insert:
        milvus_data_insert(args.data_path, args.embedding_path, args.batch_size)
    if args.search:
        milvus_data_recall(args.embedding_path, args.index)
