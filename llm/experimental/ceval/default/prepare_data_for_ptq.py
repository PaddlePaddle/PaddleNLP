# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import glob
import json
import os

import pandas as pd
from tqdm import tqdm

input_path = "../../../../dataset/ceval/dev"
output_path = "../../../../dataset/ceval_ptq_test"
os.makedirs(output_path, exist_ok=True)
quant_json = os.path.join(output_path, "quant.json")
dev_json = os.path.join(output_path, "dev.json")

with open("./subject_mapping.json", "r") as f:
    subject_mapping = json.load(f)


def generate_few_shot_prompt(k, subject, dev_df, cot=False):
    prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
    if k < 0:
        return prompt

    for i in range(k):
        prompt += format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
    return prompt


def get_data_from_file(file_path, idx=0, few_shot=False, k=-1):
    data = pd.read_csv(file_path)
    if idx >= len(data):
        return None
    line = data.iloc[idx]

    data_dict = dict()
    subject = os.path.basename(file_path).replace("_dev.csv", "")
    subject_name = subject_mapping[subject][1]
    if few_shot:
        history = generate_few_shot_prompt(k, subject_name, data, cot=True)
    else:
        history = ""

    question = format_example(line, include_answer=False, cot=False)
    example = history + question
    tgt = f'答案为{line["answer"]}\n'
    tgt += f'解释：{line["explanation"]}'
    data_dict["src"] = example
    data_dict["tgt"] = tgt
    return data_dict


def format_example(line, include_answer=True, cot=False, with_prompt=False, choices=["A", "B", "C", "D"]):
    example = line["question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'
    if include_answer:
        if cot:
            example += "\n答案：让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
        else:
            example += "\n答案：" + line["answer"] + "\n\n"
    else:
        if with_prompt is False:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
            else:
                example += "\n答案："
        else:
            if cot:
                example += "\n答案是什么？让我们一步一步思考，\n1."
            else:
                example += "\n答案是什么？ "
    return example


def get_all_data_from_file(file_path):
    data = pd.read_csv(file_path)
    data_list = []
    for row_index, line in tqdm(data.iterrows(), total=len(data)):
        data_dict = dict()
        example = line["question"]
        choices = ["A", "B", "C", "D"]
        for choice in choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        tgt = f'答案为{line["answer"]}\n'
        tgt += f'解释：{line["explanation"]}'
        data_dict["src"] = example
        data_dict["tgt"] = tgt

        data_list.append(data_dict)
    return data_dict


data_list = []
idx_dict = dict()
cnt = 0
break_cnt = len(glob.glob(os.path.join(input_path, "*.csv")))
nums = 128
while cnt < nums:
    for n in glob.glob(os.path.join(input_path, "*.csv")):
        if n in idx_dict:
            idx_dict[n] += 1
        else:
            idx_dict[n] = 0
        data = get_data_from_file(n, idx_dict[n], few_shot=True, k=3)
        if data is None:
            break_cnt -= 1
            continue
        data_list.append(data)
        cnt += 1
        if cnt >= nums:
            break
        if break_cnt <= 0:
            break
    if break_cnt <= 0:
        break

import json

with open(quant_json, "w") as f:
    for line in data_list:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


with open(dev_json, "w") as f:
    for line in data_list:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
