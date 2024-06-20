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
import os
import sys
from functools import partial

import paddle
import paddle.nn as nn
import pandas as pd

from paddlenlp.datasets import load_dataset as ppnlp_load_dataset
from paddlenlp.transformers import BertTokenizer as PPNLPBertTokenizer

CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
CONFIG_PATH = CURRENT_DIR.rsplit("/", 1)[0]
sys.path.append(CONFIG_PATH)
from models.pd_bert import BertConfig, BertForSequenceClassification  # noqa: E402


def get_data():
    def read(data_path):
        df = pd.read_csv(data_path, sep="\t")
        for _, row in df.iterrows():
            yield {"sentence": row["sentence"], "labels": row["label"]}

    def convert_example(example, tokenizer, max_length=128):
        # labels = [example["labels"]]
        # labels = np.array([example["labels"]], dtype="int64")
        example = tokenizer(example["sentence"], max_seq_len=max_length)
        return example

    tokenizer = PPNLPBertTokenizer.from_pretrained("bert-base-uncased")
    dataset_test = ppnlp_load_dataset(read, data_path="demo_sst2_sentence/demo.tsv", lazy=False)
    trans_func = partial(convert_example, tokenizer=tokenizer, max_length=128)

    dataset_test = dataset_test.map(trans_func, lazy=False)
    one_sentence = dataset_test.new_data[0]

    for k in ["input_ids", "token_type_ids"]:
        one_sentence[k] = paddle.to_tensor(one_sentence[k], dtype="int64")
        one_sentence[k] = paddle.unsqueeze(one_sentence[k], axis=0)

    return one_sentence


@paddle.no_grad()
def main():
    # 模型定义
    paddle_dump_path = "../weights/paddle_weight.pdparams"
    config = BertConfig()
    model = BertForSequenceClassification(config)
    checkpoint = paddle.load(paddle_dump_path)
    model.bert.load_dict(checkpoint)

    classifier_weights = paddle.load("../classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)

    model.eval()
    # 要预测的句子
    data = get_data()
    softmax = nn.Softmax()
    # 预测的各类别的概率值
    output = softmax(model(**data)[0]).numpy()

    # 概率值最大的类别
    class_id = output.argmax()
    # 对应的概率值
    prob = output[0][class_id]
    print(f"class_id: {class_id}, prob: {prob}")
    return output


if __name__ == "__main__":
    main()
