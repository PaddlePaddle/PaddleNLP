# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import abc
import copy
import logging
import os
from copy import deepcopy
from typing import Dict, Sequence

import paddle
from datasets import load_dataset
from paddle.io import Dataset
from tqdm import tqdm

IGNORE_INDEX = -100


# positions = "f7+l7" will intervene with the first 7 tokens and last 7 tokens
def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


# layers * intervention tokens
def get_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    last_position = kwargs["last_position"]
    if "positions" in kwargs:
        _first_n, _last_n = parse_positions(kwargs["positions"])
    else:
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
    num_interventions = kwargs["num_interventions"]

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1

    position_list = (
        [i for i in range(first_n)]
        + [i for i in range(last_position - last_n, last_position)]
        + [pad_position for _ in range(pad_amount)]
    )
    intervention_locations = [position_list] * num_interventions

    return intervention_locations


class ReftDataset(Dataset):
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        data_path,
        tokenizer,
        data_split="train",
        dataset=None,
        seed=42,
        **kwargs,
    ):
        super(ReftDataset, self).__init__()

        # setup
        self.tokenizer = tokenizer
        self.first_n, self.last_n = parse_positions(kwargs["position"])
        self.data_path = data_path
        self.data_split = data_split
        self.dataset = dataset
        self.seed = seed
        self.fields_to_pad = ["input_ids", "labels"]
        self.fields_to_mask = ["input_ids"]

        # load the dataset
        self.preprocess(kwargs)
        self.task_dataset = self.load_dataset()

        # kwargs settings
        self.postprocess(kwargs)

        # tokenize and intervene
        self.result = []
        for i, data_item in enumerate(tqdm(self.task_dataset)):
            tokenized, last_position = self.tokenize(data_item)
            tokenized = self.compute_intervention_and_subspaces(i, data_item, tokenized, last_position, **kwargs)
            self.result.append(tokenized)

    @abc.abstractmethod
    def tokenize(self, data_item, **kwargs):
        """How to tokenize a single data item. Override this function!"""
        return

    def preprocess(self, kwargs):
        """Preprocessing."""
        return

    def postprocess(self, kwargs):
        """Postprocessing."""
        return

    def __len__(self):
        return len(self.result)

    def __getitem__(self, i) -> Dict[str, paddle.Tensor]:
        return copy.deepcopy(self.result[i])

    def load_dataset(self):
        """Load the dataset (or a portion of it) from HF or a local file."""

        # load the dataset
        if self.dataset is None:
            logging.info("loading data for dataset: ", self.data_path)
            task_dataset = load_dataset("json", data_files=self.data_path, split="train")
        else:
            task_dataset = self.dataset
        self.raw_dataset = task_dataset if self.data_split != "train" else None
        return task_dataset

    def get_intervention_locations(self, **kwargs):
        return get_intervention_locations(**kwargs)

    def compute_intervention_and_subspaces(self, id: int, data_item, result: dict, last_position: int, **kwargs):
        # compute intervention locs
        intervention_locations = self.get_intervention_locations(
            last_position=last_position,
            first_n=self.first_n,
            last_n=self.last_n,
            **kwargs,
        )
        result["intervention_locations"] = intervention_locations
        result["id"] = id

        # add a single padding token BEFORE input_ids and fix everything
        for field in self.fields_to_pad:
            if field not in result:
                continue
            if field == "labels":
                result[field] = paddle.concat(
                    (
                        paddle.to_tensor(
                            [
                                IGNORE_INDEX,
                            ]
                        ),
                        result[field],
                    )
                )
            else:
                result[field] = paddle.concat(
                    (
                        paddle.to_tensor(
                            [
                                self.tokenizer.pad_token_id,
                            ]
                        ),
                        result[field],
                    )
                )
        result["intervention_locations"] = (
            paddle.to_tensor(result["intervention_locations"], dtype="int32") + 1
        ).tolist()

        # attention masks
        if len(self.fields_to_mask) == 1:
            result["attention_mask"] = (result[self.fields_to_mask[0]] != self.tokenizer.pad_token_id).astype("int32")
        else:
            for field in self.fields_to_mask:
                result[f"{field}_mask"] = (result[field] != self.tokenizer.pad_token_id).int()

        return result


# paddle version label 错开
class LoReftSupervisedDataset(ReftDataset):
    def preprocess(self, kwargs):
        self.trigger_tokens, self.num_labels = None, None
        self.task_prompt_template = "%s\n"
        self.trigger_tokens = kwargs["trigger_tokens"] if "trigger_tokens" in kwargs else None
        self.original_data_split = self.data_split
        self.data_path = os.path.join(self.data_path, self.data_split + ".json")

    def tokenize(self, data_item):
        result = {}
        base_prompt = self.task_prompt_template % (data_item["src"])
        base_input = (
            base_prompt
            + self.trigger_tokens
            + data_item["tgt"]
            # + self.tokenizer.eos_token
        )

        base_prompt_ids = self.tokenizer(
            base_prompt,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
        )["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        if self.original_data_split == "train":
            base_input_ids = self.tokenizer(
                base_input,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            )["input_ids"][0]
            output_ids = deepcopy(base_input_ids)
            # paddle 格式输入和label错开一位
            output_ids = paddle.concat([output_ids[1:], paddle.to_tensor([2])], axis=0)
            output_ids[: base_prompt_length - 1] = IGNORE_INDEX

            result["input_ids"] = base_input_ids
            result["labels"] = output_ids
        else:
            result["input_ids"] = base_prompt_ids
        last_position = base_prompt_length

        return result, last_position


class ReftDataCollator(object):
    """Collate examples for ReFT."""

    # data_collator: DataCollator
    def __init__(self, data_collator):
        self.data_collator = data_collator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, paddle.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        return batch_inputs
