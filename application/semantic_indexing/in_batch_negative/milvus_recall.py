from milvus import *

# from milvus_tool.config import MILVUS_HOST, MILVUS_PORT, top_k, search_param
from config import MILVUS_HOST, MILVUS_PORT, top_k, search_param


class RecallByMilvus():
    def __init__(self):
        self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)

    def search(self, vectors, collection_name, partition_tag=None):
        try:
            status, results = self.client.search(collection_name=collection_name, query_records=vectors, top_k=top_k,
                                                 params=search_param, partition_tag=partition_tag)
            # print(status)
            return status, results
        except Exception as e:
            print('Milvus recall error: ', e)


if __name__ == '__main__':
    import random
    client = RecallByMilvus()
    collection_name = 'test1'
    partition_tag = 'partition_3'
    embeddings = [[random.random() for _ in range(128)] for _ in range(2)]
    status, resultes = client.search(collection_name=collection_name, vectors=embeddings, partition_tag=partition_tag)
    print(status)
    print(resultes)
