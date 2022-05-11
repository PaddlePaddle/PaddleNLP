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
from paddlenlp.metrics import ChunkEvaluator
from numpy import array, float32, int32
from datasets import load_dataset

import numpy as np

SEQ_LEN = 128


def print_ret(rets, input_datas):
    for i, ret in enumerate(rets):
        print("input data:", input_datas[i])
        print("The model detects all entities:")
        for iterm in ret:
            print("entity:", iterm["entity"], "  label:", iterm["label"],
                  "  pos:", iterm["pos"])
        print("-----------------------------")


def label_pad(label_list):
    """Pad the label to the maximum length"""
    max_length = 0
    for label in label_list:
        if len(label) > max_length:
            max_length = len(label)

    max_length += 2
    if max_length > SEQ_LEN:
        max_length = SEQ_LEN

    new_label_list = []
    for label in label_list:
        if len(label) > max_length - 2:
            label = label[:max_length - 2]
        label = [0] + label + [0]
        label += [0] * (max_length - len(label))
        # print("---2-", len(label))
        new_label_list.append(label)

    return new_label_list


def test_ner_dataset(client):
    dev_ds = load_dataset("msra_ner", split="test")
    print(dev_ds["tokens"][10:15])
    return 0

    import os
    if os.environ.get('https_proxy'):
        del os.environ['https_proxy']
    if os.environ.get('http_proxy'):
        del os.environ['http_proxy']

    print("Start processing the dataset...")
    idx = 0
    batches = []
    labels = []
    batch_size = 2
    while idx < 100:
        # while idx < len(dev_ds):
        batches.append(dev_ds["tokens"][idx:idx + batch_size])
        label_list = dev_ds["ner_tags"][idx:idx + batch_size]

        label_list = label_pad(label_list)
        labels.append(np.array(label_list))
        idx += batch_size

    print("Start infer...")
    metric = ChunkEvaluator(
        label_list=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'])
    for i, data in enumerate(batches):
        ret = client.predict(feed_dict={"tokens": data})
        # print("ret:", ret)
        if i < 2:
            print_ret(eval(ret.value[0]), data)
        labels_index = eval(ret.value[1])

        # calculate metric
        import paddle
        label = paddle.to_tensor(labels[i])
        labels_index = paddle.to_tensor(labels_index)
        seq_len = [labels_index.shape[1]] * labels_index.shape[0]

        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            paddle.to_tensor(seq_len), labels_index, label)
        metric.update(num_infer_chunks.numpy(),
                      num_label_chunks.numpy(), num_correct_chunks.numpy())

    res = metric.accumulate()
    print("acc: ", res)


def init_client():
    client = PipelineClient()
    client.connect(['127.0.0.1:18090'])
    return client


def test_demo(client):
    text1 = [
        "在过去的五年中，致公党在邓小平理论指引下，遵循社会主义初级阶段的基本路线，努力实践致公党十大提出的发挥参政党职能、加强自身建设的基本任务。",
        "今年７月１日我国政府恢复对香港行使主权，标志着“一国两制”构想的巨大成功，标志着中国人民在祖国统一大业的道路上迈出了重要的一步。"
    ]
    ret = client.predict(feed_dict={"tokens": text1})
    print_ret(eval(ret.value[0]), text1)

    text2 = [[
        '中', '共', '中', '央', '致', '中', '国', '致', '公', '党', '十', '一', '大', '的',
        '贺', '词', '各', '位', '代', '表', '、', '各', '位', '同', '志', '：', '在', '中',
        '国', '致', '公', '党', '第', '十', '一', '次', '全', '国', '代', '表', '大', '会',
        '隆', '重', '召', '开', '之', '际', '，', '中', '国', '共', '产', '党', '中', '央',
        '委', '员', '会', '谨', '向', '大', '会', '表', '示', '热', '烈', '的', '祝', '贺',
        '，', '向', '致', '公', '党', '的', '同', '志', '们', '致', '以', '亲', '切', '的',
        '问', '候', '！'
    ]]
    ret = client.predict(feed_dict={"tokens": text2})
    print_ret(eval(ret.value[0]), text2)


if __name__ == "__main__":
    client = init_client()
    test_demo(client)
    # test_ner_dataset(client)
