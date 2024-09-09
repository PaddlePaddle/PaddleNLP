# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
"""Dataset class for preference training."""

from __future__ import annotations

from typing import Callable, Hashable

import numpy as np
import paddle
from paddle.io import Dataset, Subset
from typing_extensions import TypedDict  # Python 3.10+

from .base import CollatorBase, RawSample, TokenizedDataset, format_prompt, left_padding

__all__ = [
    "PromptOnlyDataset",
    "PromptOnlyCollator",
    "PromptOnlySample",
    "PromptOnlyBatch",
]


class PromptOnlySample(TypedDict, total=True):
    input_ids: paddle.Tensor  # size = (L,)


class PromptOnlyBatch(TypedDict, total=True):
    input_ids: paddle.Tensor  # size = (B, L)
    attention_mask: paddle.Tensor  # size = (B, L)


class PromptOnlyDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> PromptOnlySample:
        prompt = format_prompt(input=raw_sample["input"], eos_token=self.tokenizer.eos_token)
        input_ids = self.tokenize(prompt)
        return {
            "input_ids": input_ids,  # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, paddle.Tensor]]], dict[str, paddle.Tensor]]:
        return PromptOnlyCollator(self.tokenizer.pad_token_id)

    def _merge_raw_datasets(self, seed: int | None = None) -> Dataset[RawSample]:
        """Merge multiple raw datasets into one dataset and remove duplicates."""

        def to_hashable(raw_sample: RawSample) -> Hashable:
            input = raw_sample["input"]  # pylint: disable=redefined-builtin
            return input if isinstance(input, str) else tuple(input)

        merged = super()._merge_raw_datasets(seed)
        inputs = {to_hashable(merged[i]): i for i in range(len(merged))}
        return Subset(merged, sorted(inputs.values()))


class PromptOnlyCollator(CollatorBase):
    def __call__(self, samples: list[PromptOnlySample]) -> PromptOnlyBatch:
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [np.ones(input_id.shape, dtype=bool) for input_id in input_ids]

        input_ids = left_padding(input_ids, padding_value=self.pad_token_id)
        attention_mask = left_padding(attention_mask, padding_value=0)
        return {
            "input_ids": input_ids,  # size = (B, L)
            "attention_mask": attention_mask,  # size = (B, L)
        }
