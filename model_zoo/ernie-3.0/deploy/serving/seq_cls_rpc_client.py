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
from paddle_serving_server.pipeline import PipelineClient
from numpy import array, float32

import numpy as np


def init_client():
    client = PipelineClient()
    client.connect(['127.0.0.1:18090'])
    return client


def test_demo(client, data):
    data = np.array([x.encode('utf-8') for x in data], dtype=np.object_)
    ret = client.predict(feed_dict={"sentence": data})
    out_dict = {}
    for key, value in zip(ret.key, ret.value):
        out_dict[key] = eval(value)
    return out_dict


def test_tnews_dataset(client):
    from paddlenlp.datasets import load_dataset
    dev_ds = load_dataset('clue', "tnews", splits='dev')

    batches = []
    labels = []
    idx = 0
    batch_size = 32
    while idx < len(dev_ds):
        data = []
        label = []
        for i in range(batch_size):
            if idx + i >= len(dev_ds):
                break
            data.append(dev_ds[idx + i]["sentence"])
            label.append(dev_ds[idx + i]["label"])
        batches.append(data)
        labels.append(np.array(label))
        idx += batch_size
    """
    # data format
    ["文本1", "文本2"]

    [b"\xe6\x96\x87\xe6\x9c\xac1", b"\xe6\x96\x87\xe6\x9c\xac2"]      # after encode
    """
    accuracy = 0
    for i, data in enumerate(batches):
        data = np.array([x.encode('utf-8') for x in data], dtype=np.object_)
        ret = client.predict(feed_dict={"sentence": data})
        # print("ret:", ret)
        for index, value in zip(ret.key, ret.value):
            if index == "label":
                value = eval(value)
                # print(value, labels[i])
                accuracy += np.sum(labels[i] == value)
                break
    print("acc:", 1.0 * accuracy / len(dev_ds))


if __name__ == "__main__":
    client = init_client()
    texts = ['未来自动驾驶真的会让酒驾和疲劳驾驶成历史吗？', '黄磊接受华少快问快答，不光智商逆天，情商也不逊黄渤']
    output = test_demo(client, texts)
    print(output)
    test_tnews_dataset(client)
