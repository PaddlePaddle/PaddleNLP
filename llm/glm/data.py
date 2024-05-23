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

import numpy as np


def custom_convert_example(example, tokenizer, data_args, is_test=True):
    source = None
    title = None
    target = None
    if "source" in example and "title" in example:
        source = example["source"]
        if "title" in example.keys():
            title = example["title"]
    elif "context" in example and "answer" in example:
        source = example["context"]
        if "answer" in example.keys():
            title = example["answer"]
    else:
        assert False, "Source and title are not in the input dictionary, nor are context and answer."
    if "target" in example.keys():
        target = example["target"]
    elif "question" in example.keys():
        target = example["question"]
    example["text_a"] = "答案：" + title + "，" + "上下文：" + source
    example["text_b"] = "在已知答案的前提下，问题：" + target
    inputs = tokenizer.encode(example["text_a"], max_length=data_args.src_length - 1, truncation=True)
    inputs["input_ids"] = inputs["input_ids"][:-1] + [tokenizer.gmask_token_id] + inputs["input_ids"][-1:]
    pad_length = data_args.src_length - len(inputs["input_ids"])
    inputs["input_ids"] = np.array([inputs["input_ids"] + [tokenizer.pad_token_id] * pad_length])
    inputs["attention_mask"] = np.array([inputs["attention_mask"] + [1] + [0] * pad_length])
    sep = inputs["input_ids"].shape[1]
    inputs = tokenizer.build_inputs_for_generation(
        inputs,
        max_gen_length=data_args.tgt_length,
        targets=" " + example["text_b"] if not is_test else None,
        padding="max_length",
    )

    for input_name in inputs.keys():
        inputs[input_name] = inputs[input_name].squeeze(0)
    if is_test:
        inputs["position_ids"] = inputs["position_ids"][:, : inputs["input_ids"].shape[-1]]
        labels = tokenizer.encode(
            " " + example["text_b"], add_special_tokens=False, max_length=data_args.tgt_length - 1
        )["input_ids"]
        loss_mask = [0] * sep + [1] * len(labels) + [0] * (data_args.tgt_length - len(labels))
        labels = (
            [0] * sep
            + labels
            + [tokenizer.eop_token_id]
            + [tokenizer.pad_token_id] * (data_args.tgt_length - len(labels) - 1)
        )
        inputs["label_ids"] = labels
        inputs["loss_mask"] = loss_mask
    return inputs
