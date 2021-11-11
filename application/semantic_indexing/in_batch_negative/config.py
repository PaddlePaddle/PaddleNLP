import os
from milvus import MetricType, IndexType

MILVUS_HOST='10.21.226.173'
MILVUS_PORT = 8530

collection_param = {
    'dimension': 256,
    'index_file_size': 256,
    'metric_type': MetricType.L2
}

index_type = IndexType.IVF_FLAT
index_param = {'nlist': 1000}

top_k = 100
search_param = {'nprobe': 20}

