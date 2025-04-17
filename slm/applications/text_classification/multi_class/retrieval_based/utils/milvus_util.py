# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from config import (
    MILVUS_HOST,
    MILVUS_PORT,
    collection_param,
    index_param,
    index_type,
    search_param,
    top_k,
)
from milvus import Milvus


class VecToMilvus:
    def __init__(self):
        self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)

    def has_collection(self, collection_name):
        try:
            status, ok = self.client.has_collection(collection_name)
            return ok
        except Exception as e:
            print("Milvus has_table error:", e)

    def creat_collection(self, collection_name):
        try:
            collection_param["collection_name"] = collection_name
            status = self.client.create_collection(collection_param)
            print(status)
            return status
        except Exception as e:
            print("Milvus create collection error:", e)

    def create_index(self, collection_name):
        try:
            status = self.client.create_index(collection_name, index_type, index_param)
            print(status)
            return status
        except Exception as e:
            print("Milvus create index error:", e)

    def has_partition(self, collection_name, partition_tag):
        try:
            status, ok = self.client.has_partition(collection_name, partition_tag)
            return ok
        except Exception as e:
            print("Milvus has partition error: ", e)

    def delete_partition(self, collection_name, partition_tag):
        try:
            status = self.client.drop_collection(collection_name)
            return status
        except Exception as e:
            print("Milvus has partition error: ", e)

    def create_partition(self, collection_name, partition_tag):
        try:
            status = self.client.create_partition(collection_name, partition_tag)
            print("create partition {} successfully".format(partition_tag))
            return status
        except Exception as e:
            print("Milvus create partition error: ", e)

    def insert(self, vectors, collection_name, ids=None, partition_tag=None):
        try:
            if not self.has_collection(collection_name):
                self.creat_collection(collection_name)
                self.create_index(collection_name)
                print("collection info: {}".format(self.client.get_collection_info(collection_name)[1]))
            if (partition_tag is not None) and (not self.has_partition(collection_name, partition_tag)):
                self.create_partition(collection_name, partition_tag)
            status, ids = self.client.insert(
                collection_name=collection_name, records=vectors, ids=ids, partition_tag=partition_tag
            )
            self.client.flush([collection_name])
            print(
                "Insert {} entities, there are {} entities after insert data.".format(
                    len(ids), self.client.count_entities(collection_name)[1]
                )
            )
            return status, ids
        except Exception as e:
            print("Milvus insert error:", e)


class RecallByMilvus:
    def __init__(self):
        self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)

    def search(self, vectors, collection_name, partition_tag=None):
        try:
            status, results = self.client.search(
                collection_name=collection_name,
                query_records=vectors,
                top_k=top_k,
                params=search_param,
                partition_tag=partition_tag,
            )
            return status, results
        except Exception as e:
            print("Milvus recall error: ", e)
