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
sys.path.append('../../cblue/')

import json
import numpy as np
from numpy import array, float32, int32, float64
from paddle_serving_server.pipeline import PipelineClient
from utils import NERChunkEvaluator


def print_ret(rets, input_datas):
    for i, ret in enumerate(rets):
        print("input data:", input_datas[i])
        print("The model detects all entities:")
        for iterm in ret:
            sid, eid = iterm["position"]
            print("entity:", input_datas[i][sid - 1:eid], ", type:",
                  iterm["type"], ", position:", (sid - 1, eid))
        print("-----------------------------")


def label_pad(label_list, preds, pad=32):
    """Pad the label to the maximum length"""
    new_label_list = []
    for label, pred in zip(label_list, preds):
        seq_len = len(pred)
        if len(label) > seq_len - 2:
            label = label[:seq_len - 2]
        label = [pad] + label + [pad]
        label += [pad] * (seq_len - len(label))
        new_label_list.append(label)
    return new_label_list


def test_ner_dataset(client):
    from paddlenlp.datasets import load_dataset
    import paddle

    _, dev_ds, _ = load_dataset("cblue", "CMeEE", split=["dev"])

    import os
    if os.environ.get('https_proxy'):
        del os.environ['https_proxy']
    if os.environ.get('http_proxy'):
        del os.environ['http_proxy']

    print("Start infer...")
    metric = NERChunkEvaluator([[
        'B-bod', 'I-bod', 'E-bod', 'S-bod', 'B-dis', 'I-dis', 'E-dis', 'S-dis',
        'B-pro', 'I-pro', 'E-pro', 'S-pro', 'B-dru', 'I-dru', 'E-dru', 'S-dru',
        'B-ite', 'I-ite', 'E-ite', 'S-ite', 'B-mic', 'I-mic', 'E-mic', 'S-mic',
        'B-equ', 'I-equ', 'E-equ', 'S-equ', 'B-dep', 'I-dep', 'E-dep', 'S-dep',
        'O'
    ], ['B-sym', 'I-sym', 'E-sym', 'S-sym', 'O']])
    idx = 0
    batch_size = 32
    max_len = len(dev_ds) - 1
    while idx < max_len:
        end_idx = idx + batch_size if idx + batch_size < max_len else max_len
        data = [x['text'] for x in dev_ds[idx:end_idx]]
        ret = client.predict(feed_dict={"tokens": data})
        if ret.err_no != 0:
            raise ValueError("err_no", ret.err_no, "err_msg: ", ret.err_msg)
        if idx < batch_size * 2:
            print_ret(json.loads(ret.value[0]), data)

        # calculate metric
        lengths = [len(x) + 1 for x in data]
        preds = [
            paddle.to_tensor(json.loads(ret.value[2])),
            paddle.to_tensor(json.loads(ret.value[1]))
        ]
        label_list = [[x['labels'][0] for x in dev_ds[idx:end_idx]],
                      [x['labels'][1] for x in dev_ds[idx:end_idx]]]
        label_list = [
            paddle.to_tensor(label_pad(label_list[0], preds[0], 32)),
            paddle.to_tensor(label_pad(label_list[1], preds[1], 4))
        ]

        correct = metric.compute(lengths, preds, label_list)
        metric.update(correct)
        idx += batch_size

    res = metric.accumulate()
    print("f1: ", res[0])


def init_client():
    client = PipelineClient()
    client.connect(['127.0.0.1:18090'])
    return client


def test_demo(client):
    text = [
        "研究证实，细胞减少与肺内病变程度及肺内炎性病变吸收程度密切相关。",
        "可为不规则发热、稽留热或弛张热，但以不规则发热为多，可能与患儿应用退热药物导致热型不规律有关。"
    ]
    ret = client.predict(feed_dict={"tokens": text})
    print(ret)
    value = json.loads(ret.value[0])
    print_ret(value, text)


if __name__ == "__main__":
    client = init_client()
    test_demo(client)
