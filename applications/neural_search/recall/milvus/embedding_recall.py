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

import numpy as np
from milvus_insert import VecToMilvus
import random
from tqdm import tqdm
from milvus_recall import RecallByMilvus
import time

embeddings = np.load('corpus_embedding.npy')
print(embeddings.shape)

embedding_ids = [i for i in range(embeddings.shape[0])]
print(len(embedding_ids))
client = VecToMilvus()
collection_name = 'literature_search'
partition_tag = 'partition_2'
data_size = len(embedding_ids)
client = RecallByMilvus()
embeddings = embeddings[np.arange(1, 2)]
time_start = time.time()  #开始计时
status, resultes = client.search(collection_name=collection_name,
                                 vectors=embeddings,
                                 partition_tag=partition_tag)
time_end = time.time()  #结束计时

sum_t = time_end - time_start  #运行所花时间
print('time cost', sum_t, 's')
print(status)
print(resultes)
