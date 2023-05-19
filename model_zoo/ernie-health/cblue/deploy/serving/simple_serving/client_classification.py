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
parser.add_argument("--dataset", required=True, type=str, help="The dataset name for the simple seving")
parser.add_argument(
    "--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization."
)
parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for predicting.")
args = parser.parse_args()

url = "http://0.0.0.0:8189/models/cblue_cls"
headers = {"Content-Type": "application/json"}

TEXT = {
    "kuake-qic": ["心肌缺血如何治疗与调养呢？", "什么叫痔核脱出？什么叫外痔？"],
    "kuake-qtr": [["儿童远视眼怎么恢复视力", "远视眼该如何保养才能恢复一些视力"], ["抗生素的药有哪些", "抗生素类的药物都有哪些？"]],
    "kuake-qqr": [["茴香是发物吗", "茴香怎么吃？"], ["气的胃疼是怎么回事", "气到胃痛是什么原因"]],
    "chip-ctc": ["(1)前牙结构发育不良：釉质发育不全、氟斑牙、四环素牙等；", "怀疑或确有酒精或药物滥用史；"],
    "chip-sts": [["糖尿病能吃减肥药吗？能治愈吗？", "糖尿病为什么不能吃减肥药"], ["H型高血压的定义", "WHO对高血压的最新分类定义标准数值"]],
    "chip-cdn-2c": [["1型糖尿病性植物神经病变", " 1型糖尿病肾病IV期"], ["髂腰肌囊性占位", "髂肌囊肿"]],
}

if __name__ == "__main__":
    args.dataset = args.dataset.lower()
    input_data = TEXT[args.dataset]
    texts = []
    text_pairs = []
    for data in input_data:
        if len(data) == 2:
            text_pairs.append(data[1])
        texts.append(data[0])
    data = {
        "data": {"text": texts, "text_pair": text_pairs if len(text_pairs) > 0 else None},
        "parameters": {"max_seq_len": args.max_seq_len, "batch_size": args.batch_size},
    }
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.text)
