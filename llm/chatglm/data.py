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


def convert_chatglm_example(example, tokenizer, data_args, is_test=True):

    if "content" in example:
        query = example["content"]
        response = example["summary"]
    elif "instruction" in example:
        query = example["instruction"]
        response = example["output"]
    elif "src" in example:
        query = example["src"][0] if isinstance(example["src"], list) else example["src"]
        response = example["tgt"][0] if isinstance(example["tgt"], list) else example["tgt"]
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
            **tokenizer(prompt, max_length=data_args.src_length, truncation=True, truncation_side="left"),
            "labels": tokenizer(response, max_length=data_args.tgt_length, truncation=True, truncation_side="right")[
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

        # shift labels
        input_ids, labels = input_ids[:-1], labels[1:]

        attention_mask = attention_mask[..., :-1, :-1]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return inputs


def convert_chatglm_v2_example(example, tokenizer, data_args, is_test=True):
    if "content" in example:
        query = example["content"]
        response = example["summary"]
    elif "instruction" in example:
        query = example["instruction"]
        response = example["output"]
    elif "src" in example:
        query = example["src"][0] if isinstance(example["src"], list) else example["src"]
        response = example["tgt"][0] if isinstance(example["tgt"], list) else example["tgt"]
    history = example.get("history", None)

    prompt = tokenizer.build_prompt(query, history)

    if is_test:
        input_ids = tokenizer(prompt, max_length=data_args.src_length, truncation=True, truncation_side="left")[
            "input_ids"
        ]
        labels = tokenizer(response, max_length=data_args.tgt_length, truncation=True, truncation_side="right")[
            "input_ids"
        ]
        # pass in position_ids explicitly because of left padding
        position_ids = list(range(len(input_ids)))
        return {"input_ids": input_ids, "labels": labels, "position_ids": position_ids}
    else:
        a_ids = tokenizer(text=prompt, add_special_tokens=True, truncation=True, max_length=data_args.src_length - 1)[
            "input_ids"
        ]
        b_ids = tokenizer(text=response, add_special_tokens=False, truncation=True, max_length=data_args.tgt_length)[
            "input_ids"
        ]
        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
        labels = [-100] * context_length + b_ids + [tokenizer.eos_token_id]

        # shift input_ids and labels
        input_ids, labels = input_ids[:-1], labels[1:]
        # pass in position_ids explicitly because of left padding
        position_ids = list(range(len(input_ids)))
        return {"input_ids": input_ids, "labels": labels, "position_ids": position_ids}
