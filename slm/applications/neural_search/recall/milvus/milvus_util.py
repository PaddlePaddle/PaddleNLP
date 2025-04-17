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


import numpy as np
from config import (
    MILVUS_HOST,
    MILVUS_PORT,
    data_dim,
    index_config,
    search_params,
    top_k,
)
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

fmt = "\n=== {:30} ===\n"
text_max_len = 1000
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=text_max_len),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=data_dim),
]
schema = CollectionSchema(fields, "Neural Search Index")


class VecToMilvus:
    def __init__(self):
        print(fmt.format("start connecting to Milvus"))
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = None

    def has_collection(self, collection_name):
        try:
            has = utility.has_collection(collection_name)
            print(f"Does collection {collection_name} exist in Milvus: {has}")
            return has
        except Exception as e:
            print("Milvus has_table error:", e)

    def creat_collection(self, collection_name):
        try:
            print(fmt.format("Create collection {}".format(collection_name)))
            self.collection = Collection(collection_name, schema, consistency_level="Strong")
        except Exception as e:
            print("Milvus create collection error:", e)

    def drop_collection(self, collection_name):
        try:
            utility.drop_collection(collection_name)
        except Exception as e:
            print("Milvus delete collection error:", e)

    def create_index(self, index_name):
        try:
            print(fmt.format("Start Creating index"))
            self.collection.create_index(index_name, index_config)
            print(fmt.format("Start loading"))
            self.collection.load()
        except Exception as e:
            print("Milvus create index error:", e)

    def has_partition(self, partition_tag):
        try:
            result = self.collection.has_partition(partition_tag)
            return result
        except Exception as e:
            print("Milvus has partition error: ", e)

    def create_partition(self, partition_tag):
        try:
            self.collection.create_partition(partition_tag)
            print("create partition {} successfully".format(partition_tag))
        except Exception as e:
            print("Milvus create partition error: ", e)

    def insert(self, entities, collection_name, index_name, partition_tag=None):
        try:
            if not self.has_collection(collection_name):
                self.creat_collection(collection_name)
                self.create_index(index_name)
            else:
                self.collection = Collection(collection_name)
            if (partition_tag is not None) and (not self.has_partition(partition_tag)):
                self.create_partition(partition_tag)

            self.collection.insert(entities, partition_name=partition_tag)
            print(f"Number of entities in Milvus: {self.collection.num_entities}")  # check the num_entites
        except Exception as e:
            print("Milvus insert error:", e)


class RecallByMilvus:
    def __init__(self):
        print(fmt.format("start connecting to Milvus"))
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = None

    def get_collection(self, collection_name):
        try:
            print(fmt.format("Connect collection {}".format(collection_name)))
            self.collection = Collection(collection_name)
        except Exception as e:
            print("Milvus create collection error:", e)

    def search(self, vectors, embedding_name, collection_name, partition_names=[], output_fields=[]):
        try:
            self.get_collection(collection_name)
            result = self.collection.search(
                vectors,
                embedding_name,
                search_params,
                limit=top_k,
                partition_names=partition_names,
                output_fields=output_fields,
            )
            return result
        except Exception as e:
            print("Milvus recall error: ", e)


if __name__ == "__main__":
    print(fmt.format("Start inserting entities"))
    rng = np.random.default_rng(seed=19530)
    num_entities = 3000
    entities = [
        # provide the pk field because `auto_id` is set to False
        [i for i in range(num_entities)],
        ["第{}个样本".format(i) for i in range(num_entities)],  # field text, only supports list
        rng.random((num_entities, data_dim)),  # field embeddings, supports numpy.ndarray and list
    ]
    print(entities[-1].shape)
    collection_name = "test1"
    partition_tag = "partition_1"
    embedding_name = "embeddings"
    client = VecToMilvus()
    client.insert(
        collection_name=collection_name, entities=entities, index_name=embedding_name, partition_tag=partition_tag
    )
    print(fmt.format("Start searching entities"))
    vectors_to_search = entities[-1][-2:]
    recall_client = RecallByMilvus()
    result = recall_client.search(
        vectors_to_search,
        embedding_name,
        collection_name,
        partition_names=[partition_tag],
        output_fields=["pk", "text"],
    )
    for hits in result:
        for hit in hits:
            print(f"hit: {hit}, random field: {hit.entity.get('text')}")
