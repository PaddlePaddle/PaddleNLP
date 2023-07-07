# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import json
from dataclasses import dataclass

import numpy as np
import paddle

from paddlenlp.data import DataCollatorForSeq2Seq

IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def reader(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            yield json_line


def convert_example(example, tokenizer, data_args, is_test=False):
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
    except:
        print(example["context"])
        print(example["question"])
        print(example["answers"])
        print(example["answer_starts"])
        print(example["is_impossible"])

    input_seq = f"answer: {answer} context: {context} </s>"
    output_seq = f"question: {question} </s>"

    source_tokenized = tokenizer(
        input_seq,
        return_tensors="pd",
        max_length=data_args.src_length,
        truncation=True,
    )

    source_input_ids_len = (
        source_tokenized["input_ids"].not_equal(paddle.to_tensor(tokenizer.pad_token_id)).sum().item()
    )

    example_tokenized = tokenizer(
        input_seq + output_seq,
        return_tensors="pd",
        max_length=data_args.src_length + data_args.tgt_length,
        padding=False,
        truncation=True,
    )

    input_ids = example_tokenized["input_ids"][0]
    labels = copy.deepcopy(input_ids)
    labels[:source_input_ids_len] = IGNORE_INDEX

    if is_test:
        return dict(
            input_ids=source_tokenized["input_ids"][0],
            labels=labels,
        )

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def custom_instruction_convert_example(
    example, tokenizer, data_args, is_test=False, benchmark=False, model_max_length=512
):
    """
    Convert an example into necessary features.
    """

    if benchmark:
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        if example.get("input", "") != "":
            input_seq = prompt_input.format_map(example)
        else:
            input_seq = prompt_no_input.format_map(example)

        output_seq = example["output"] + tokenizer.eos_token
    else:
        instruction = ""
        input = ""
        output = ""
        if "instruction" in example and "output" in example:
            instruction = example["instruction"]
            output = example["output"]
        else:
            assert False, "instruction and output are not in the input dictionary."
        if "input" in example["input"]:
            input = example["input"]

        input_seq = instruction + input
        output_seq = output + tokenizer.eos_token

    # To compatible with compile training mode in benchmark, input will be pad to fix length
    source_tokenized = tokenizer(
        input_seq,
        return_tensors="pd",
        max_length=data_args.src_length if not benchmark else model_max_length,
        truncation=True,
    )

    source_input_ids_len = (
        source_tokenized["input_ids"].not_equal(paddle.to_tensor(tokenizer.pad_token_id)).sum().item()
    )

    total_length = data_args.src_length + data_args.tgt_length

    example_tokenized = tokenizer(
        input_seq + output_seq,
        return_tensors="pd",
        max_length=total_length if not benchmark else model_max_length,
        truncation=True,
    )

    input_ids = example_tokenized["input_ids"][0]
    labels = copy.deepcopy(input_ids)
    labels[:source_input_ids_len] = IGNORE_INDEX

    if is_test:
        return dict(
            input_ids=source_tokenized["input_ids"][0],
            labels=labels,
        )

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def left_padding(inputs, pad_id, max_length=-1):
    for ids in inputs:
        max_length = max(max_length, len(ids))

    def extend_max_lenth(value, max_length, to_pad_id):
        return [to_pad_id] * (max_length - len(value)) + value

    def extend_filed(values, max_length, to_pad_id):
        res = []
        for value in values:
            res.append(extend_max_lenth(value.tolist(), max_length, to_pad_id))
        return res

    res = extend_filed(inputs, max_length, pad_id)
    return paddle.to_tensor(res)


@dataclass
class DataCollatorForSupervisedDataset(DataCollatorForSeq2Seq):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, features, return_tensors=None):
        # Deep copy to avoid modifying features in-place
        batch = copy.deepcopy(features)
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in batch] if "labels" in batch[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            # Note(gongenlei): In pipeline, max_label_length = self.max_length
            if self.padding == "max_length" and self.max_length is not None:
                max_label_length = self.max_length
            else:
                max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in batch:
                remainder = [IGNORE_INDEX] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
            return_attention_mask=self.return_attention_mask,
        )

        return batch
