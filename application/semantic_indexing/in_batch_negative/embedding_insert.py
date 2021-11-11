import numpy as np
from milvus_insert import VecToMilvus
import random
from tqdm import tqdm 

embeddings=np.load('corpus_embedding.npy') 
print(embeddings.shape)

embedding_ids = [i for i in range(embeddings.shape[0])]
# embed_list=[embeddings[i].tolist() for i in range(embeddings.shape[0])]
# print(len(embed_list))
print(len(embedding_ids))
client = VecToMilvus()
collection_name = 'wanfang1'
partition_tag = 'partition_2'
data_size=len(embedding_ids)
batch_size=100000
for i in tqdm(range(0,data_size,batch_size)):
    batch_emb=embeddings[np.arange(i,i+batch_size)]
    status, ids = client.insert(collection_name=collection_name, vectors=batch_emb.tolist(), ids=embedding_ids[i:i+batch_size],partition_tag=partition_tag)
    # print(status)
    # print(ids)

# ids = [random.randint(0, 1000) for _ in range(100)]
# embeddings = [[random.random() for _ in range(128)] for _ in range(100)]
# print(len(ids))
# print(len(embeddings))