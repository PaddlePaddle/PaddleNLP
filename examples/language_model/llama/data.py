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
from dataclasses import dataclass
from typing import Dict, List

import paddle

from paddlenlp.data import Pad
from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase

IGNORE_INDEX = 0  # TODO: Temporarily set to 0, fix after ParallelCrossEntropy support -100


def convert_example(example, tokenizer, data_args, is_test=True):
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

    source_tokenized = tokenizer(
        example["text_a"],
        return_tensors="pd",
        max_length=data_args.src_length,
        truncation=True,
    )

    source_input_ids_len = (
        source_tokenized["input_ids"].not_equal(paddle.to_tensor(tokenizer.pad_token_id)).sum().item()
    )

    example_tokenized = tokenizer(
        example["text_a"] + example["text_b"],
        return_tensors="pd",
        max_length=data_args.src_length + data_args.tgt_length,
        truncation=True,
    )

    input_ids = example_tokenized["input_ids"][0]
    labels = copy.deepcopy(input_ids)
    labels[:source_input_ids_len] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def pad_sequence(inputs, pad_index=0):
    sequences = [inp.numpy() for inp in inputs]
    outputs = Pad(pad_val=pad_index)(sequences)
    output_tensor = paddle.to_tensor(outputs)
    return output_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PretrainedTokenizerBase

    def __call__(self, features: List[Dict]) -> Dict[str, paddle.Tensor]:

        input_ids, labels = tuple([feature[key] for feature in features] for key in ("input_ids", "labels"))
        input_ids = pad_sequence(input_ids, pad_index=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, pad_index=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.not_equal(paddle.to_tensor(self.tokenizer.pad_token_id)),
        )
