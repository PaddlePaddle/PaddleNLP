from milvus import *

# from milvus_tool.config import MILVUS_HOST, MILVUS_PORT, collection_param, index_type, index_param
from config import MILVUS_HOST, MILVUS_PORT, collection_param, index_type, index_param


class VecToMilvus():
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
            collection_param['collection_name'] = collection_name
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

    def create_partition(self, collection_name, partition_tag):
        try:
            status = self.client.create_partition(collection_name, partition_tag)
            print('create partition {} successfully'.format(partition_tag))
            return status
        except Exception as e:
            print('Milvus create partition error: ', e)

    def insert(self, vectors, collection_name, ids=None, partition_tag=None):
        try:
            if not self.has_collection(collection_name):
                self.creat_collection(collection_name)
                self.create_index(collection_name)
                print('collection info: {}'.format(self.client.get_collection_info(collection_name)[1]))
            if (partition_tag is not None) and (not self.has_partition(collection_name, partition_tag)):
                self.create_partition(collection_name, partition_tag)
            status, ids = self.client.insert(collection_name=collection_name, records=vectors, ids=ids,
                                             partition_tag=partition_tag)
            self.client.flush([collection_name])
            print('Insert {} entities, there are {} entities after insert data.'.format(len(ids), self.client.count_entities(collection_name)[1]))
            return status, ids
        except Exception as e:
            print("Milvus insert error:", e)


if __name__ == '__main__':
    import random

    client = VecToMilvus()
    collection_name = 'test1'
    partition_tag = 'partition_1'
    ids = [random.randint(0, 1000) for _ in range(100)]
    embeddings = [[random.random() for _ in range(128)] for _ in range(100)]
    status, ids = client.insert(collection_name=collection_name, vectors=embeddings, ids=ids,partition_tag=partition_tag)
    print(status)
    # print(ids)
