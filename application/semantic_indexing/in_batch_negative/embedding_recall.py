import numpy as np
from milvus_insert import VecToMilvus
import random
from tqdm import tqdm 
from milvus_recall import RecallByMilvus
import time

embeddings=np.load('corpus_embedding.npy') 
print(embeddings.shape)
embedding_ids = [i for i in range(embeddings.shape[0])]
# embed_list=[embeddings[i].tolist() for i in range(embeddings.shape[0])]
# print(len(embed_list))
print(len(embedding_ids))
# client = VecToMilvus()
collection_name = 'wanfang1'
partition_tag = 'partition_2'
data_size=len(embedding_ids)
batch_size=100
# for i in tqdm(range(0,data_size,batch_size)):
#     status, ids = client.insert(collection_name=collection_name, vectors=embed_list[i:i+batch_size], ids=embedding_ids[i:i+20],partition_tag=partition_tag)
#     print(status)
#     print(ids)

# ids = [random.randint(0, 1000) for _ in range(100)]
# embeddings = [[random.random() for _ in range(128)] for _ in range(100)]
# print(len(ids))
# print(len(embeddings))

client = RecallByMilvus()
embeddings = embeddings[np.arange(0,1)]
time_start = time.time() #开始计时
status, resultes = client.search(collection_name=collection_name, vectors=embeddings, partition_tag=partition_tag)
time_end = time.time()    #结束计时
sum_t=time_end - time_start   #运行所花时间
print('time cost', sum_t, 's')
print(status)
print(resultes)