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

# convert data for eval
# AIME 2024
dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
dataset.to_json("data/aime2024/dev.json", force_ascii=False)

# MATH-500
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
dataset.to_json("data/math500/dev.json", force_ascii=False)

# PaddlePaddle/GSM8K_distilled_zh
dataset = load_dataset("PaddlePaddle/GSM8K_distilled_zh")
dataset["train"].to_json("data/gsm8k_distilled_zh/GSM8K_distilled_zh-train.json", force_ascii=False)
dataset["test"].to_json("data/gsm8k_distilled_zh/GSM8K_distilled_zh-test.json", force_ascii=False)

# make data for sft
def process_data(example):
    src = example.get("question_zh", "")
    content = example.get("deepseek_r1_response_zh", "")
    reasoning_content = example.get("deepseek_r1_reasoning_zh", "")
    tgt = reasoning_content + content
    return {"src": src, "tgt": tgt}


paddlenlp_datatset = deepcopy(dataset)
paddlenlp_datatset["train"] = paddlenlp_datatset["train"].map(
    process_data, remove_columns=paddlenlp_datatset["train"].column_names
)
paddlenlp_datatset["test"] = paddlenlp_datatset["test"].map(
    process_data, remove_columns=paddlenlp_datatset["test"].column_names
)
paddlenlp_datatset["train"].to_json("data/gsm8k_distilled_zh_sft/GSM8K_distilled_zh-train.json", force_ascii=False)
paddlenlp_datatset["test"].to_json("data/gsm8k_distilled_zh_sft/GSM8K_distilled_zh-test.json", force_ascii=False)
