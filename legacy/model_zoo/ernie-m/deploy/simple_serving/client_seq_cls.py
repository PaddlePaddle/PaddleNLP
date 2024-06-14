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
from datasets import load_dataset

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--language", required=True, type=str, help="The language for the simple seving")
parser.add_argument("--max_seq_len", default=256, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for predicting.")
args = parser.parse_args()
# yapf: enable

url = "http://0.0.0.0:8189/models/ernie_m_cls"
headers = {"Content-Type": "application/json"}


if __name__ == "__main__":
    examples = load_dataset("xnli", args.language, split="validation")[:10]
    texts = [text for text in examples["premise"]]
    text_pairs = [text for text in examples["hypothesis"]]

    data = {
        "data": {"text": texts, "text_pair": text_pairs},
        "parameters": {"max_seq_len": args.max_seq_len, "batch_size": args.batch_size},
    }
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.text)
