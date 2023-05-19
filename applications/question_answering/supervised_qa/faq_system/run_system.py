# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


import time

import numpy as np
import pandas as pd
from data import gen_id2corpus
from milvus_util import RecallByMilvus
from paddle_serving_server.pipeline import PipelineClient


def search_in_milvus(text_embedding, query_text):
    collection_name = "faq_system"
    partition_tag = "partition_1"
    client = RecallByMilvus()
    start_time = time.time()
    status, results = client.search(
        collection_name=collection_name, vectors=text_embedding, partition_tag=partition_tag
    )
    end_time = time.time()
    print("Search milvus time cost is {} seconds ".format(end_time - start_time))

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
    df = pd.DataFrame(list_data, columns=["query_text", "text", "distance"])
    df = df.sort_values(by="distance", ascending=True)
    df.to_csv("data/recall_predict.csv", columns=["text", "distance"], sep="\t", header=None, index=False)


if __name__ == "__main__":
    client = PipelineClient()
    client.connect(["127.0.0.1:8080"])
    list_data = ["嘉定区南翔镇实行双门长制“门长”要求落实好哪些工作？"]
    feed = {}
    for i, item in enumerate(list_data):
        feed[str(i)] = item
    start_time = time.time()
    ret = client.predict(feed_dict=feed)
    end_time = time.time()
    print("Extract feature time to cost :{} seconds".format(end_time - start_time))
    result = np.array(eval(ret.value[0]))
    search_in_milvus(result, list_data[0])
