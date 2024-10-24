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

import sys
import time

import numpy as np
import pandas as pd
from paddle_serving_server.pipeline import PipelineClient

sys.path.append("./recall/milvus")  # noqa: E402
from config import collection_name, embedding_name, partition_tag  # noqa: E402
from milvus_util import RecallByMilvus  # noqa: E402


def recall_result(list_data):
    client = PipelineClient()
    client.connect(["127.0.0.1:8080"])
    feed = {}
    for i, item in enumerate(list_data):
        feed[str(i)] = item
    start_time = time.time()
    ret = client.predict(feed_dict=feed)
    end_time = time.time()
    print("Extract feature time to cost :{} seconds".format(end_time - start_time))
    result = np.array(eval(ret.value[0]))
    return result


def search_in_milvus(embeddings, query_text):
    recall_client = RecallByMilvus()
    start_time = time.time()
    results = recall_client.search(
        embeddings, embedding_name, collection_name, partition_names=[partition_tag], output_fields=["pk", "text"]
    )
    end_time = time.time()
    print("Search milvus time cost is {} seconds ".format(end_time - start_time))
    list_data = []
    for line in results:
        for item in line:
            # idx = item.id
            distance = item.distance
            text = item.entity.get("text")
            list_data.append([query_text, text, distance])
    df = pd.DataFrame(list_data, columns=["query_text", "text", "distance"])
    df.to_csv("recall_result.csv", index=False)
    return df


def rerank(df):
    client = PipelineClient()
    client.connect(["127.0.0.1:8089"])
    list_data = []
    for index, row in df.iterrows():
        example = {"query": row["query_text"], "title": row["text"]}
        list_data.append(example)
    feed = {}
    for i, item in enumerate(list_data):
        feed[str(i)] = str(item)

    start_time = time.time()
    ret = client.predict(feed_dict=feed)
    end_time = time.time()
    print("time to cost :{} seconds".format(end_time - start_time))
    result = np.array(eval(ret.value[0]))
    df["distance"] = result
    df = df.sort_values(by=["distance"], ascending=False)
    df.to_csv("rank_result.csv", index=False)


if __name__ == "__main__":
    list_data = ["中西方语言与文化的差异"]
    result = recall_result(list_data)
    df = search_in_milvus(result, list_data[0])
    rerank(df)
