# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import json

import numpy as np


def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line.strip())


def convert_example(example, tokenizer, data_args, is_test=True):
    query = example["content"]
    response = example["summary"]
    history = example.get("history", None)

    if history is None or len(history) == 0:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, old_response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, old_response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

    # dataset for evaluation
    if is_test:
        inputs = {
            **tokenizer(prompt, max_length=data_args.src_length, truncation=True, padding="max_length"),
            "labels": tokenizer(response, max_length=data_args.tgt_length, truncation=True, padding="max_length")[
                "input_ids"
            ],
        }
    # dataset for training
    else:
        src_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=data_args.src_length - 1,
            truncation=True,
            truncation_side="left",
        )["input_ids"]
        tgt_ids = tokenizer(
            response,
            add_special_tokens=False,
            max_length=data_args.tgt_length - 2,
            truncation=True,
            truncation_side="right",
        )["input_ids"]

        input_ids = tokenizer.build_inputs_with_special_tokens(src_ids, tgt_ids)

        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1

        attention_mask = np.tri(len(input_ids), len(input_ids))
        attention_mask[:, :context_length] = 1
        attention_mask = attention_mask[None, :, :]
        attention_mask = (attention_mask < 0.5).astype("int64")

        labels = [-100] * context_length + input_ids[mask_position + 1 :]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return inputs
