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

from typing import Callable

import numpy as np
import paddle
from typing_extensions import TypedDict  # Python 3.10+

from .base import (
    CollatorBase,
    RawSample,
    TokenizedDataset,
    format_prompt,
    right_padding,
)

__all__ = [
    "PreferenceDataset",
    "PreferenceCollator",
    "PreferenceSample",
    "PreferenceBatch",
]


class PreferenceSample(TypedDict, total=True):
    better_input_ids: paddle.Tensor  # size = (L,)
    worse_input_ids: paddle.Tensor  # size = (L,)


class PreferenceBatch(TypedDict, total=True):
    better_input_ids: paddle.Tensor  # size = (B, L)
    better_attention_mask: paddle.Tensor  # size = (B, L)

    worse_input_ids: paddle.Tensor  # size = (B, L)
    worse_attention_mask: paddle.Tensor  # size = (B, L)


class PreferenceDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample, index=0) -> PreferenceSample:
        prompt = format_prompt(input=raw_sample["input"], eos_token=self.tokenizer.eos_token)
        better_answer = raw_sample["answer"]
        worse_answer = raw_sample["other_answer"]
        better = raw_sample["better"]
        if not better:
            better_answer, worse_answer = worse_answer, better_answer

        better_input_ids = self.tokenize(prompt + better_answer + self.tokenizer.eos_token)
        worse_input_ids = self.tokenize(prompt + worse_answer + self.tokenizer.eos_token)

        if (
            better_input_ids.shape == worse_input_ids.shape
            and np.all(np.equal(better_input_ids, worse_input_ids)).item()
        ):
            # raise ValueError(
            import warnings

            warnings.warn(
                "Two responses get the same `input_ids` after tokenization.\n\n"
                f"Prompt: {prompt}\n\n"
                f"Better answer: {better_answer}\n\n"
                f"Worse answer: {worse_answer}",
            )
        return {
            "better_input_ids": better_input_ids,  # size = (L,)
            "worse_input_ids": worse_input_ids,  # size = (L,)
        }

    def get_collator(self) -> Callable[[list[dict[str, paddle.Tensor]]], dict[str, paddle.Tensor]]:
        return PreferenceCollator(self.tokenizer.pad_token_id)


class PreferenceCollator(CollatorBase):
    def __call__(self, samples: list[PreferenceSample]) -> PreferenceBatch:
        input_ids = [sample["better_input_ids"] for sample in samples] + [
            sample["worse_input_ids"] for sample in samples
        ]  # size = (2 * B, L)
        attention_mask = [np.ones(input_id.shape, dtype=bool) for input_id in input_ids]  # size = (2 * B, L)

        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)  # size = (2 * B, L)
        attention_mask = right_padding(attention_mask, padding_value=0)  # size = (2 * B, L)

        bs = len(samples)
        (better_input_ids, worse_input_ids,) = (  # size = (B, L)  # size = (B, L)
            input_ids[:bs],
            input_ids[bs:],
        )
        (better_attention_mask, worse_attention_mask,) = (  # size = (B, L)  # size = (B, L)
            attention_mask[:bs],
            attention_mask[bs:],
        )

        return {
            "better_input_ids": better_input_ids,  # size = (B, L)
            "better_attention_mask": better_attention_mask,  # size = (B, L)
            "worse_input_ids": worse_input_ids,  # size = (B, L)
            "worse_attention_mask": worse_attention_mask,  # size = (B, L)
        }
