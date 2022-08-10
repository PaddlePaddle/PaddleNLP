from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from paddle_serving_server.pipeline import PipelineClient
from data import gen_id2corpus
from milvus_util import RecallByMilvus


def recall_result(list_data):
    client = PipelineClient()
    client.connect(['127.0.0.1:8080'])
    feed = {}
    for i, item in enumerate(list_data):
        feed[str(i)] = item
    start_time = time.time()
    ret = client.predict(feed_dict=feed)
    end_time = time.time()
    print("Extract feature time to cost :{} seconds".format(end_time -
                                                            start_time))
    result = np.array(eval(ret.value[0]))
    return result


def search_in_milvus(text_embedding, query_text, id2corpus):
    collection_name = 'literature_search'
    partition_tag = 'partition_2'
    client = RecallByMilvus()
    start_time = time.time()
    status, results = client.search(collection_name=collection_name,
                                    vectors=text_embedding,
                                    partition_tag=partition_tag)
    end_time = time.time()
    print('Search milvus time cost is {} seconds '.format(end_time -
                                                          start_time))
    list_data = []
    for line in results:
        for item in line:
            idx = item.id
            distance = item.distance
            text = id2corpus[idx]
            list_data.append([query_text, text, distance])
    df = pd.DataFrame(list_data, columns=['query_text', 'text', 'distance'])
    df.to_csv('recall_result.csv', index=False)
    return df


def rerank(df):
    client = PipelineClient()
    client.connect(['127.0.0.1:8089'])
    list_data = []
    for index, row in df.iterrows():
        example = {"query": row['query_text'], "title": row['text']}
        list_data.append(example)
    feed = {}
    for i, item in enumerate(list_data):
        feed[str(i)] = str(item)

    start_time = time.time()
    ret = client.predict(feed_dict=feed)
    end_time = time.time()
    print("time to cost :{} seconds".format(end_time - start_time))
    result = np.array(eval(ret.value[0]))
    df['distance'] = result
    df = df.sort_values(by=["distance"], ascending=False)
    df.to_csv('rank_result.csv', index=False)


if __name__ == "__main__":
    list_data = ["中西方语言与文化的差异"]
    corpus_file = "milvus/milvus_data.csv"
    id2corpus = gen_id2corpus(corpus_file)
    result = recall_result(list_data)
    df = search_in_milvus(result, list_data[0], id2corpus)
    rerank(df)
