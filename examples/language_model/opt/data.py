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
from __future__ import annotations

import paddle


def convert_example(
    example,
    tokenizer,
    max_source_length,
    max_target_length,
    is_train=False,
):
    """
    Convert an example into necessary features.
    """
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    context = example["context"]
    question = example["question"]
    try:
        answer = example["answers"][0]
    except Exception:
        print(example["context"])
        print(example["question"])
        print(example["answers"])
        print(example["answer_starts"])
        print(example["is_impossible"])

    input_seq = f"answer: {answer} context: {context} </s>"
    output_seq = f"question: {question} </s>"

    outputs = tokenizer(
        output_seq,
        padding="max_length",
        max_length=max_target_length,
        truncation_strategy="longest_first",
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    inputs = tokenizer(
        input_seq,
        max_seq_len=max_source_length,
        truncation_strategy="longest_first",
        return_attention_mask=True,
        return_length=False,
    )

    final = {}
    for k in outputs.keys():
        final[k] = inputs[k] + outputs[k]
        if k == "input_ids":
            final["labels"] = [tokenizer.pad_token_id] * len(inputs["input_ids"]) + outputs[k]
    return {k: paddle.to_tensor(v, dtype="int64") for k, v in final.items()}
