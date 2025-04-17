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

import argparse
import json

import requests

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--prob_limit", default=0.5, type=float, help="The limitation of probability for the label.")
args = parser.parse_args()
# yapf: enable

url = "http://0.0.0.0:8189/models/cls_hierarchical"
headers = {"Content-Type": "application/json"}

if __name__ == "__main__":
    texts = [
        "请问木竭胶囊能同高血压药、氨糖同时服吗？",
        "低压100*高压140*头涨，想吃点降压药。谢谢！",
        "脑穿通畸形易发人群有哪些",
        "幼儿乱吃丙硫氧嘧啶片怎么办，我也不知道她吃了几片",
        "如果是可以降血糖的话,血糖值7点多的大概需要吃几个疗程?",
    ]
    data = {
        "data": {
            "text": texts,
        },
        "parameters": {"max_seq_len": args.max_seq_len, "batch_size": args.batch_size, "prob_limit": args.prob_limit},
    }
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.text)
