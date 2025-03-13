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

from copy import deepcopy

from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main")
paddlenlp_dataset = deepcopy(dataset["train"])
paddlenlp_dataset = paddlenlp_dataset.rename_column("question", "src")
paddlenlp_dataset = paddlenlp_dataset.rename_column("answer", "tgt")
paddlenlp_dataset.to_json("data/gsm8k/train.json", force_ascii=False)
paddlenlp_dataset = deepcopy(dataset["test"])
paddlenlp_dataset = paddlenlp_dataset.rename_column("question", "src")
paddlenlp_dataset = paddlenlp_dataset.rename_column("answer", "tgt")
paddlenlp_dataset.to_json("data/gsm8k/dev.json", force_ascii=False)


dataset = load_dataset("camel-ai/gsm8k_distilled")["train"]
paddlenlp_dataset = deepcopy(dataset)
paddlenlp_dataset = paddlenlp_dataset.rename_column("problem", "src")
paddlenlp_dataset = paddlenlp_dataset.rename_column("reasoning_solution", "tgt")
paddlenlp_dataset.to_json("data/gsm8k_distilled/train.json", force_ascii=False)


dataset = load_dataset("meta-math/GSM8K_zh")["train"]
paddlenlp_dataset = deepcopy(dataset)
paddlenlp_dataset.to_json("data/gsm8k_zh/train.jsonl", force_ascii=False)


dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
paddlenlp_dataset = deepcopy(dataset)
paddlenlp_dataset = paddlenlp_dataset.rename_column("problem", "src")
paddlenlp_dataset = paddlenlp_dataset.rename_column("solution", "tgt")
paddlenlp_dataset.to_json("data/aime2024/dev.json", force_ascii=False)


dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
paddlenlp_dataset = deepcopy(dataset)
paddlenlp_dataset = paddlenlp_dataset.rename_column("problem", "src")
paddlenlp_dataset = paddlenlp_dataset.rename_column("solution", "tgt")
paddlenlp_dataset.to_json("data/math500/dev.json", force_ascii=False)
