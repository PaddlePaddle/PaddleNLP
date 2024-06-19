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

from milvus import IndexType, MetricType

MILVUS_HOST = "10.21.226.173"
MILVUS_PORT = 8530

output_emb_size = 0

collection_param = {
    "dimension": output_emb_size if output_emb_size > 0 else 768,
    "index_file_size": 256,
    "metric_type": MetricType.L2,
}

index_type = IndexType.FLAT
index_param = {"nlist": 1000}

top_k = 20
search_param = {"nprobe": 20}

collection_name = "text"
partition_tag = "partition_2"
