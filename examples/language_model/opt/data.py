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

from dataclasses import dataclass
from typing import Dict, List

import paddle

from paddlenlp.data import Pad
from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase


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
    answer = (example.get("answers", []) or ["there is no answer"])[0]

    input_seq = f"context: {context}. question: {question}. answer: "
    output_seq = answer

    # 1. tokenize input-tokens
    input_tokens = tokenizer.tokenize(input_seq)[:max_source_length]

    # 2. tokenize output tokens
    output_tokens = tokenizer.tokenize(output_seq)[:max_target_length]

    # 3. concat the inputs
    tokens = input_tokens + output_tokens

    labels = [-100] * len(input_tokens) + tokenizer.convert_tokens_to_ids(output_tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def pad_sequence(inputs, pad_index=0):
    outputs = Pad(pad_val=pad_index)(inputs)
    output_tensor = paddle.to_tensor(outputs)
    return output_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PretrainedTokenizerBase = None
    inf_tensor = paddle.to_tensor(-1e4, dtype="float32")
    zero_tensor = paddle.to_tensor(0, dtype="float32")

    def __call__(self, features: List[Dict]) -> Dict[str, paddle.Tensor]:

        input_ids, labels = tuple([feature[key] for feature in features] for key in ("input_ids", "labels"))
        input_ids = pad_sequence(input_ids, pad_index=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, pad_index=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=paddle.where(input_ids != self.tokenizer.pad_token_id, self.inf_tensor, self.zero_tensor),
        )
