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

MILVUS_HOST = "10.21.226.175"
MILVUS_PORT = 8530
data_dim = 256
top_k = 100
collection_name = "literature_search"
partition_tag = "partition_2"
embedding_name = "embeddings"

index_config = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1000},
}

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": top_k},
}
