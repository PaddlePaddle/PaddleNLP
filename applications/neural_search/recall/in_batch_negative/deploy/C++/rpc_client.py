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

from paddle_serving_client import Client
from paddlenlp.transformers import AutoTokenizer


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    pad_to_max_seq_len=True):
    list_input_ids = []
    list_token_type_ids = []
    for text in example:
        encoded_inputs = tokenizer(text=text,
                                   max_seq_len=max_seq_length,
                                   pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        list_input_ids.append(input_ids)
        list_token_type_ids.append(token_type_ids)
    return list_input_ids, list_token_type_ids


# 启动python客户端
endpoint_list = ['127.0.0.1:9393']
client = Client()
client.load_client_config('serving_client')
client.connect(endpoint_list)
feed_names = client.feed_names_
fetch_names = client.fetch_names_
print(feed_names)
print(fetch_names)

# 创建tokenizer
tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')
max_seq_len = 64

# 数据预处理

list_data = ['国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据.', '面向生态系统服务的生态系统分类方案研发与应用']
# for i in range(5):
#     list_data.extend(list_data)
# print(len(list_data))
examples = convert_example(list_data, tokenizer, max_seq_length=max_seq_len)
print(examples)

feed_dict = {}
feed_dict['input_ids'] = np.array(examples[0])
feed_dict['token_type_ids'] = np.array(examples[1])

print(feed_dict['input_ids'].shape)
print(feed_dict['token_type_ids'].shape)
# batch设置为True表示的是批量预测
b_start = time.time()
result = client.predict(feed=feed_dict, fetch=fetch_names, batch=True)
b_end = time.time()
print("time to cost :{} seconds".format(b_end - b_start))
print(result)
