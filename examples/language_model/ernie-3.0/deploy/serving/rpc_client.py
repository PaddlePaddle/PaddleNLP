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
try:
    from paddle_serving_server.pipeline import PipelineClient
except ImportError:
    from paddle_serving_server.pipeline import PipelineClient
import numpy as np
import requests
import json

from paddlenlp.datasets import load_dataset
from numpy import array, float32

client = PipelineClient()
client.connect(['127.0.0.1:18090'])

dev_ds = load_dataset('clue', "tnews", splits='dev')

batches = []
labels = []
idx = 0
batch_size = 2
while idx < len(dev_ds):
    datas = []
    label = []
    for i in range(batch_size):
        if idx + i >= len(dev_ds):
            break
        datas.append(dev_ds[idx + i]["sentence"])
        label.append(dev_ds[idx + i]["label"])
    batches.append(datas)
    labels.append(np.array(label))
    idx += batch_size

accuracy = 0
for i, data in enumerate(batches):
    data = np.array([x.encode('utf-8') for x in data], dtype=np.object_)
    ret = client.predict(feed_dict={"sentence" : data}, fetch=["test"])
    # print("ret:", ret)
    for index, value in zip(ret.key, ret.value):
        if index == "index":
            value = eval(value)
            # print(value, labels[i])
            accuracy += np.sum(labels[i] == value)
            break
print("acc:", 1.0*accuracy/len(dev_ds))
