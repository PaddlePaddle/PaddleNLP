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
parser.add_argument("--max_length", default=512, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU/CPU for predicting.")
parser.add_argument("--threshold", default=0.8, type=float, help="Probability threshold for prediction.")
args = parser.parse_args()
# yapf: disable

url = "http://0.0.0.0:8190/models/utc"
headers = {"Content-Type": "application/json"}

if __name__ == "__main__":
    data = {
        "data": [{"text_a": "借款贷款。"}],
        "parameters": {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "prob_limit": args.threshold,
            "choices": ["正向", "负向"],
        }
    }
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.text)
