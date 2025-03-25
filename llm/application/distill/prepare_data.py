# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from copy import deepcopy

from datasets import load_dataset

# convert data for distill
# GSM8K
dataset = load_dataset("meta-math/GSM8K_zh")["train"]
dataset.to_json("data/gsm8k_zh/GSM8K_zh.jsonl", force_ascii=False)

# PaddlePaddle/GSM8K_distilled_zh
dataset = load_dataset("PaddlePaddle/GSM8K_distilled_zh")
dataset["train"].to_json("data/gsm8k_distilled_zh/GSM8K_distilled_zh-train.json", force_ascii=False)
dataset["test"].to_json("data/gsm8k_distilled_zh/GSM8K_distilled_zh-test.json", force_ascii=False)


# make data for sft
def process_data_zh(example):
    src = example.get("question_zh", "")
    content = example.get("deepseek_r1_response_zh", "")
    reasoning_content = example.get("deepseek_r1_reasoning_zh", "")
    tgt = reasoning_content + content
    return {"src": src, "tgt": tgt}


def process_data_en(example):
    src = example.get("question", "")
    content = example.get("deepseek_r1_response", "")
    reasoning_content = example.get("deepseek_r1_reasoning", "")
    tgt = reasoning_content + content
    return {"src": src, "tgt": tgt}


# construct Chinese sft dataset
paddlenlp_datatset = deepcopy(dataset)
paddlenlp_datatset["train"] = paddlenlp_datatset["train"].map(
    process_data_zh, remove_columns=paddlenlp_datatset["train"].column_names
)
paddlenlp_datatset["test"] = paddlenlp_datatset["test"].map(
    process_data_zh, remove_columns=paddlenlp_datatset["test"].column_names
)
paddlenlp_datatset["train"].to_json("data/gsm8k_distilled_zh_sft/train.json", force_ascii=False)
paddlenlp_datatset["test"].to_json("data/gsm8k_distilled_zh_sft/dev.json", force_ascii=False)

# construct English sft dataset
paddlenlp_datatset = deepcopy(dataset)
paddlenlp_datatset["train"] = paddlenlp_datatset["train"].map(
    process_data_en, remove_columns=paddlenlp_datatset["train"].column_names
)
paddlenlp_datatset["test"] = paddlenlp_datatset["test"].map(
    process_data_en, remove_columns=paddlenlp_datatset["test"].column_names
)
paddlenlp_datatset["train"].to_json("data/gsm8k_distilled_en_sft/train.json", force_ascii=False)
paddlenlp_datatset["test"].to_json("data/gsm8k_distilled_en_sft/dev.json", force_ascii=False)
