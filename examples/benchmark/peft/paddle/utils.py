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

import copy
from dataclasses import dataclass

import numpy as np

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.trainer import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

IGNORE_INDEX = -100


class CustomTrainer(Trainer):
    total_observed_tokens = 0.0

    def training_step(self, model, inputs):
        input_ids = inputs["input_ids"]
        self.total_observed_tokens += float(input_ids.shape[0] * input_ids.shape[1])
        return super().training_step(model, inputs)


class ProfilerCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, prof):
        self.prof = prof
        self.prof.start()

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        self.prof.stop()
        self.prof.summary()


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
            return_attention_mask=True,
        )

        return batch
