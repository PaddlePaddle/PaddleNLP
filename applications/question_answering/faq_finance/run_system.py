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


def search_in_milvus(text_embedding, query_text):
    collection_name = 'faq_finance'
    partition_tag = 'partition_1'
    client = RecallByMilvus()
    start_time = time.time()
    status, results = client.search(collection_name=collection_name,
                                    vectors=text_embedding,
                                    partition_tag=partition_tag)
    end_time = time.time()
    print('Search milvus time cost is {} seconds '.format(end_time -
                                                          start_time))

    corpus_file = "data/qa_pair.csv"
    id2corpus = gen_id2corpus(corpus_file)
    list_data = []
    for line in results:
        for item in line:
            idx = item.id
            distance = item.distance
            text = id2corpus[idx]
            print(text, distance)
            list_data.append([query_text, text, distance])
    df = pd.DataFrame(list_data, columns=['query_text', 'text', 'distance'])
    df = df.sort_values(by="distance", ascending=True)
    df.to_csv('data/recall_predict.csv',
              columns=['text', 'distance'],
              sep='\t',
              header=None,
              index=False)


if __name__ == "__main__":
    client = PipelineClient()
    client.connect(['127.0.0.1:8080'])
    list_data = ["买了社保，是不是就不用买商业保险了？"]
    feed = {}
    for i, item in enumerate(list_data):
        feed[str(i)] = item
    start_time = time.time()
    ret = client.predict(feed_dict=feed)
    end_time = time.time()
    print("Extract feature time to cost :{} seconds".format(end_time -
                                                            start_time))
    result = np.array(eval(ret.value[0]))
    search_in_milvus(result, list_data[0])
