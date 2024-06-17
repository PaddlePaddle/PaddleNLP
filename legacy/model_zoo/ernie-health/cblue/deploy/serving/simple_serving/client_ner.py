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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization."
)
parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU/CPU for predicting.")
args = parser.parse_args()

url = "http://0.0.0.0:8189/models/cblue_ner"
headers = {"Content-Type": "application/json"}

if __name__ == "__main__":
    texts = ["研究证实，细胞减少与肺内病变程度及肺内炎性病变吸收程度密切相关。", "可为不规则发热、稽留热或弛张热，但以不规则发热为多，可能与患儿应用退热药物导致热型不规律有关。"]
    texts = [[x.lower() for x in text] for text in texts]
    data = {
        "data": {
            "text": texts,
        },
        "parameters": {"max_seq_len": args.max_seq_len, "batch_size": args.batch_size, "is_split_into_words": True},
    }
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.text)
