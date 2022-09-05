# coding:utf-8
# pylint: disable=doc-string-missing
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import requests
import json

from paddle_serving_client.httpclient import HttpClient
from scipy.special import expit
from paddlenlp.transformers import AutoTokenizer
import paddlenlp as ppnlp


def convert_example(example, tokenizer, max_seq_length=512):

    query, title = example["query"], example["title"]
    encoded_inputs = tokenizer(text=query,
                               text_pair=title,
                               max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    return input_ids, token_type_ids


# 启动python客户端
endpoint_list = ['127.0.0.1:8600']
client = HttpClient()
client.load_client_config('serving_client')
client.connect(endpoint_list)
feed_names = client.feed_names_
fetch_names = client.fetch_names_

# 创建tokenizer
tokenizer = AutoTokenizer.from_pretrained('rocketqa-base-cross-encoder')
max_seq_len = 64

# 数据预处理
list_data = [{
    "query": "加强科研项目管理有效促进医学科研工作",
    "title": "科研项目管理策略科研项目,项目管理,实施,必要性,策略"
}]

input_ids, token_type_ids = [], []
for example in list_data:
    input_id, token_type_id = convert_example(example,
                                              tokenizer,
                                              max_seq_length=max_seq_len)
    input_ids.append(input_id)
    token_type_ids.append(token_type_id)

feed_dict = {}
feed_dict['input_ids'] = np.array(input_ids)
feed_dict['token_type_ids'] = np.array(token_type_ids)
# batch设置为True表示的是批量预测
b_start = time.time()
result = client.predict(feed=feed_dict, fetch=fetch_names, batch=True)
b_end = time.time()
print(result)
print("time to cost :{} seconds".format(b_end - b_start))
score = result.outputs[0].tensor[0].float_data
sim_score = expit(np.array(score))[1]
print(sim_score)
