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

from milvus import *
from config import MILVUS_HOST, MILVUS_PORT, top_k, search_param


class RecallByMilvus():

    def __init__(self):
        self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)

    def search(self, vectors, collection_name, partition_tag=None):
        try:
            status, results = self.client.search(
                collection_name=collection_name,
                query_records=vectors,
                top_k=top_k,
                params=search_param,
                partition_tag=partition_tag)
            return status, results
        except Exception as e:
            print('Milvus recall error: ', e)


if __name__ == '__main__':
    import random
    client = RecallByMilvus()
    collection_name = 'literature_search'
    partition_tag = 'partition_3'
    embeddings = [[random.random() for _ in range(128)] for _ in range(2)]
    status, resultes = client.search(collection_name=collection_name,
                                     vectors=embeddings,
                                     partition_tag=partition_tag)
    print(status)
    print(resultes)
