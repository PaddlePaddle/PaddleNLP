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
from numpy import array, float32, int32, float64

import numpy as np
import json


def print_ret(rets, input_data):
    for i, ret in enumerate(rets):
        print("input data:", input_data[i])
        print("The model detects all entities:")
        for iterm in ret:
            print("entity:", iterm["entity"], "  label:", iterm["label"],
                  "  pos:", iterm["pos"])
        print("-----------------------------")


def label_pad(label_list, preds):
    """Pad the label to the maximum length"""
    new_label_list = []
    for label, pred in zip(label_list, preds):
        seq_len = len(pred)
        if len(label) > seq_len - 2:
            label = label[:seq_len - 2]
        label = [0] + label + [0]
        label += [0] * (seq_len - len(label))
        new_label_list.append(label)
    return new_label_list


def test_ner_dataset(client):
    from paddlenlp.metrics import ChunkEvaluator
    from datasets import load_dataset
    import paddle

    dev_ds = load_dataset("msra_ner", split="test")

    import os
    if os.environ.get('https_proxy'):
        del os.environ['https_proxy']
    if os.environ.get('http_proxy'):
        del os.environ['http_proxy']

    print("Start infer...")
    metric = ChunkEvaluator(
        label_list=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'])
    idx = 0
    batch_size = 32
    max_len = len(dev_ds["tokens"]) - 1
    while idx < max_len:
        end_idx = idx + batch_size if idx + batch_size < max_len else max_len
        data = dev_ds["tokens"][idx:end_idx]
        ret = client.predict(feed_dict={"tokens": data})
        if ret.err_no != 0:
            raise ValueError("err_no", ret.err_no, "err_msg: ", ret.err_msg)
        # print("ret:", ret)
        if idx < batch_size * 2:
            print_ret(json.loads(ret.value[0]), data)

        # calculate metric
        preds = json.loads(ret.value[1])
        label_list = dev_ds["ner_tags"][idx:end_idx]
        label_list = label_pad(label_list, preds)
        label_list = paddle.to_tensor(label_list)
        preds = paddle.to_tensor(preds)
        seq_len = [preds.shape[1]] * preds.shape[0]

        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            paddle.to_tensor(seq_len), preds, label_list)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(),
                      num_correct_chunks.numpy())
        idx += batch_size
        print(idx)

    res = metric.accumulate()
    print("acc: ", res)


def init_client():
    client = PipelineClient()
    client.connect(['127.0.0.1:18090'])
    return client


def test_demo(client):
    text1 = [
        "北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。",
        "原产玛雅故国的玉米，早已成为华夏大地主要粮食作物之一。",
    ]
    ret = client.predict(feed_dict={"tokens": text1})
    value = json.loads(ret.value[0])
    print_ret(value, text1)

    text2 = [
        [
            '从', '首', '都', '利', '隆', '圭', '乘', '车', '向', '湖', '边', '小', '镇',
            '萨', '利', '马', '进', '发', '时', '，', '不', '到', '１', '０', '０', '公',
            '里', '的', '道', '路', '上', '坑', '坑', '洼', '洼', '，', '又', '逢', '阵',
            '雨', '迷', '蒙', '，', '令', '人', '不', '时', '发', '出', '路', '难', '行',
            '的', '慨', '叹', '。'
        ],
    ]
    ret = client.predict(feed_dict={"tokens": text2})
    value = json.loads(ret.value[0])
    print_ret(value, text2)


if __name__ == "__main__":
    client = init_client()
    test_demo(client)
