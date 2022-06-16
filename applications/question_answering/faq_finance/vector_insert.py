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

import random

from tqdm import tqdm
import numpy as np

from milvus_util import VecToMilvus


def vector_insert(file_path):
    embeddings = np.load(file_path)
    print(embeddings.shape)
    embedding_ids = [i for i in range(embeddings.shape[0])]
    print(len(embedding_ids))
    client = VecToMilvus()
    collection_name = 'faq_finance'
    partition_tag = 'partition_1'
    data_size = len(embedding_ids)
    batch_size = 100000
    for i in tqdm(range(0, data_size, batch_size)):
        cur_end = i + batch_size
        if (cur_end > data_size):
            cur_end = data_size
        batch_emb = embeddings[np.arange(i, cur_end)]
        status, ids = client.insert(collection_name=collection_name,
                                    vectors=batch_emb.tolist(),
                                    ids=embedding_ids[i:i + batch_size],
                                    partition_tag=partition_tag)


if __name__ == "__main__":
    file_path = 'corpus_embedding.npy'
    vector_insert(file_path)
