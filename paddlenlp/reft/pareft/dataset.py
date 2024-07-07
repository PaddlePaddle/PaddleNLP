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

IGNORE_INDEX = -100

no_header_prompt_template = """\
### Instruction:
%s

### Response:
"""

prompt_input = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

prompt_no_input = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""

import abc
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import datasets
import paddle
import transformers
from datasets import load_dataset
from paddle.io import Dataset
from tqdm import tqdm

import paddlenlp

# from transformers import DataCollator
# from paddlenlp.data import DataCollator


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


def get_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_position = kwargs["last_position"]
    if "positions" in kwargs:
        _first_n, _last_n = parse_positions(kwargs["positions"])
    else:
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
    num_interventions = kwargs["num_interventions"]
    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_position
    if share_weights or (first_n == 0 or last_n == 0):
        position_list = (
            [i for i in range(first_n)]
            + [i for i in range(last_position - last_n, last_position)]
            + [pad_position for _ in range(pad_amount)]
        )
        intervention_locations = [position_list] * num_interventions
    else:
        left_pad_amount = _first_n - first_n
        right_pad_amount = _last_n - last_n
        left_intervention_locations = [i for i in range(first_n)] + [pad_position for _ in range(left_pad_amount)]
        right_intervention_locations = [i for i in range(last_position - last_n, last_position)] + [
            pad_position for _ in range(right_pad_amount)
        ]
        # after padding, there could be still length diff, we need to do another check
        left_len = len(left_intervention_locations)
        right_len = len(right_intervention_locations)
        if left_len > right_len:
            right_intervention_locations += [pad_position for _ in range(left_len - right_len)]
        else:
            left_intervention_locations += [pad_position for _ in range(right_len - left_len)]
        intervention_locations = [left_intervention_locations] * (num_interventions // 2) + [
            right_intervention_locations
        ] * (num_interventions // 2)

    return intervention_locations


class ReftDataset(Dataset):
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        task: str,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_split="train",
        dataset=None,
        seed=42,
        max_n_example=None,
        **kwargs,
    ):
        super(ReftDataset, self).__init__()
        # result = defaultdict(list)

        # setup
        self.tokenizer = tokenizer
        self.first_n, self.last_n = parse_positions(kwargs["position"])
        self.task = task
        self.data_path = data_path
        self.data_split = data_split
        self.dataset = dataset
        self.seed = seed
        self.max_n_example = max_n_example
        self.pad_mode = "first"
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
            print("loading data for dataset: ", self.data_path)
            if self.data_path is None:
                task_dataset = load_dataset(self.task, split=self.data_split)
            elif self.data_path.endswith(".json"):
                task_dataset = load_dataset("json", data_files=self.data_path, split="train")
            else:
                task_dataset = load_dataset(self.task, self.data_path, split=self.data_split)
        else:
            task_dataset = self.dataset

        # select n random examples if specificed
        if self.max_n_example is not None:
            task_dataset = task_dataset.shuffle(seed=self.seed)
            task_dataset = task_dataset.select(range(self.max_n_example))

        # save raw_dataset pointer for access raw strings
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
            pad_mode=self.pad_mode,
            **kwargs,
        )
        result["intervention_locations"] = intervention_locations
        result["id"] = id
        # return result
        # print("result", result)

        # add a single padding token BEFORE input_ids and fix everything
        if self.pad_mode == "first":
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
                # torch.IntTensor(result["intervention_locations"]) + 1
                paddle.to_tensor(result["intervention_locations"], dtype="int32")
                + 1
            ).tolist()
        elif self.pad_mode == "last":
            for field in self.fields_to_pad:
                if field not in result:
                    continue
                if field == "labels":
                    result[field] = paddle.concat(
                        (
                            result[field],
                            paddle.to_tensor(
                                [
                                    IGNORE_INDEX,
                                ]
                            ),
                        )
                    )
                else:
                    result[field] = paddle.concat(
                        (
                            result[field],
                            paddle.to_tensor(
                                [
                                    self.tokenizer.pad_token_id,
                                ]
                            ),
                        )
                    )

        # attention masks
        if len(self.fields_to_mask) == 1:
            # result["attention_mask"] = (
            #     result[self.fields_to_mask[0]] != self.tokenizer.pad_token_id
            # ).int()
            result["attention_mask"] = (result[self.fields_to_mask[0]] != self.tokenizer.pad_token_id).astype("int32")
        else:
            for field in self.fields_to_mask:
                result[f"{field}_mask"] = (result[field] != self.tokenizer.pad_token_id).int()

        # subspaces
        if "subspaces" in data_item:
            num_interventions = kwargs["num_interventions"]
            share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
            if share_weights:
                num_interventions = num_interventions // 2
            # we now assume each task has a constant subspaces
            _subspaces = [data_item["subspaces"]] * num_interventions
            result["subspaces"] = _subspaces

        return result


class ReftClassificationDataset(ReftDataset):
    """
    A ReftClassificationDataset only contains a single text field
    that we tokenize, intervene on a prefix + suffix of, and
    compute subspace settings for. This is intended for classification
    tasks.

    Remember to pass in the input_field and label_field as kwargs.
    """

    def preprocess(self, kwargs):
        self.input_field = kwargs["input_field"]
        self.label_field = kwargs["label_field"]

    def tokenize(self, data_item):
        result = {}

        # input
        input_ids = self.tokenizer(
            data_item[self.input_field],
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        base_prompt_length = len(input_ids)
        last_position = base_prompt_length - 1
        result["input_ids"] = input_ids

        # labels
        if self.label_field == self.input_field:
            result["labels"] = input_ids.clone()
        elif self.label_field is not None:
            labels = self.tokenizer(
                data_item[self.label_field],
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"][0]
            result["labels"] = labels

        return result, last_position


class ReftGenerationDataset(ReftDataset):
    """
    A ReftGenerationDataset contains an instruction and a
    completion for each data item. We intervene on a prefix + suffix
    of *only the instruction*. This is suitable for generation tasks
    where you don't want inference overhead during decoding.

    Remember to pass in the prompt_field and completion_field as kwargs.
    """

    def preprocess(self, kwargs):
        self.prompt_field = kwargs["prompt_field"]
        self.completion_field = kwargs["completion_field"]

    def tokenize(self, data_item):
        result = {}

        # prompt
        prompt_ids = self.tokenizer(
            data_item[self.prompt_field],
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        base_prompt_length = len(prompt_ids)
        last_position = base_prompt_length - 1

        # input
        full_input = data_item[self.prompt_field] + data_item[self.completion_field] + self.tokenizer.eos_token
        input_ids = self.tokenizer(
            full_input,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        result["input_ids"] = input_ids

        # labels
        output_ids = copy.deepcopy(input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX
        result["labels"] = output_ids

        return result, last_position


class ReftSupervisedDataset(ReftDataset):
    """
    Alpaca-style supervised dataset. We intervene on a prefix + suffix
    of the input. This is suitable for supervised fine-tuning tasks.

    Remember to pass in the input_field, output_field, and instruction_field as kwargs.
    """

    def preprocess(self, kwargs):
        self.input_field = kwargs["input_field"]
        self.output_field = kwargs["output_field"]
        self.instruction_field = kwargs["instruction_field"]

    def tokenize(self, data_item):
        result = {}

        # prompt
        if self.input_field not in data_item or data_item[self.input_field] == "":
            base_prompt = prompt_no_input % (data_item[self.instruction_field])
        else:
            base_prompt = prompt_input % (
                data_item[self.instruction_field],
                data_item[self.input_field],
            )
        prompt_ids = self.tokenizer(
            base_prompt,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        base_prompt_length = len(prompt_ids)
        last_position = base_prompt_length - 1

        # input
        base_input = base_prompt + data_item[self.output_field] + self.tokenizer.eos_token
        input_ids = self.tokenizer(
            base_input,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        result["input_ids"] = input_ids

        # labels
        output_ids = copy.deepcopy(input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX
        result["labels"] = output_ids

        return result, last_position


def make_last_position_supervised_chat_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    model,
    inputs,
    outputs,
    num_interventions=1,
    nonstop=False,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
    for i in range(len(inputs)):
        _input = inputs[i]
        _output = outputs[i]

        base_prompt = _input
        base_input = base_prompt + _output
        if not nonstop:
            base_input += tokenizer.eos_token

        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_input_ids = tokenizer(
            base_input,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX

        all_base_input_ids.append(base_input_ids)
        all_intervention_locations.append([[base_prompt_length - 1]] * num_interventions)
        all_output_ids.append(output_ids)

    train_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": all_base_input_ids,
            "intervention_locations": all_intervention_locations,
            "labels": all_output_ids,
        }
    )

    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_last_position_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    model,
    inputs,
    outputs,
    num_interventions=1,
    nonstop=False,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
    for i in range(len(inputs)):
        _input = inputs[i]
        _output = outputs[i]

        base_prompt = _input
        base_input = base_prompt + _output
        # if not nonstop:
        #     base_input += tokenizer.eos_token

        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
        )["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_input_ids = tokenizer(
            base_input,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
        )["input_ids"][0]
        # paddle要求错开一位
        output_ids = copy.deepcopy(base_input_ids)
        output_ids = paddle.concat([output_ids[1:], paddle.to_tensor([2])], axis=0)
        output_ids[: base_prompt_length - 1] = IGNORE_INDEX

        all_base_input_ids.append(base_input_ids)
        all_intervention_locations.append([[base_prompt_length - 1]] * num_interventions)
        all_output_ids.append(output_ids)

    train_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": all_base_input_ids,
            "intervention_locations": all_intervention_locations,
            "labels": all_output_ids,
        }
    )

    # data_collator_fn = transformers.DataCollatorForSeq2Seq(
    #     tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
    # )
    data_collator_fn = paddlenlp.data.DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_multiple_position_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    model,
    inputs,
    outputs,
    positions="f1+l1",
    num_interventions=1,
    nonstop=False,
    share_weights=False,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    first_n, last_n = parse_positions(positions)

    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
    for i in range(len(inputs)):
        _input = inputs[i]
        _output = outputs[i]

        base_prompt = _input
        base_input = base_prompt + _output
        if not nonstop:
            base_input += tokenizer.eos_token

        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_input_ids = tokenizer(
            base_input,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX

        intervention_locations = get_intervention_locations(
            last_position=base_prompt_length,
            first_n=first_n,
            last_n=last_n,
            pad_mode="last",
            num_interventions=num_interventions,
            share_weights=share_weights,
        )

        all_base_input_ids.append(base_input_ids)
        all_intervention_locations.append(intervention_locations)
        all_output_ids.append(output_ids)

    train_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": all_base_input_ids,
            "intervention_locations": all_intervention_locations,
            "labels": all_output_ids,
        }
    )

    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class ReftRewardDataset(ReftDataset):
    def preprocess(self, kwargs):
        self.conv_A_field = kwargs["conv_A_field"]
        self.conv_B_field = kwargs["conv_B_field"]
        self.prompt_field = kwargs["prompt_field"] if "prompt_field" in kwargs else None
        self.conv_A_reward_field = kwargs["conv_A_reward_field"] if "conv_A_reward_field" in kwargs else None
        self.conv_B_reward_field = kwargs["conv_B_reward_field"] if "conv_B_reward_field" in kwargs else None
        self.fields_to_pad = [
            "chosen_output",
            "rejected_output",
        ]  # pad both chosen and rejected with dummy tok
        self.fields_to_mask = [
            "chosen_output",
            "rejected_output",
        ]  # -> chosen_output_mask, rejected_output_mask

    def tokenize(self, data_item):
        result = {}

        # generate prompt format
        if self.prompt_field is not None:
            data_item[self.conv_A_field] = [
                {"role": "user", "content": data_item[self.prompt_field]},
                {"role": "assistant", "content": data_item[self.conv_A_field]},
            ]
            data_item[self.conv_B_field] = [
                {"role": "user", "content": data_item[self.prompt_field]},
                {"role": "assistant", "content": data_item[self.conv_B_field]},
            ]
        chosen_output = self.tokenizer.apply_chat_template(
            data_item[self.conv_A_field], tokenize=False, add_generation_prompt=False
        ).replace(self.tokenizer.bos_token, "")
        rejected_output = self.tokenizer.apply_chat_template(
            data_item[self.conv_B_field], tokenize=False, add_generation_prompt=False
        ).replace(self.tokenizer.bos_token, "")

        # reward
        if self.conv_A_reward_field is not None:
            result["chosen_reward"] = data_item[self.conv_A_reward_field]
            result["rejected_reward"] = data_item[self.conv_B_reward_field]

            # swap so that chosen is better
            if result["chosen_reward"] < result["rejected_reward"]:
                chosen_output, rejected_output = rejected_output, chosen_output
                result["chosen_reward"], result["rejected_reward"] = (
                    result["rejected_reward"],
                    result["chosen_reward"],
                )

        # tokenize
        chosen_ids = self.tokenizer(
            chosen_output,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        rejected_ids = self.tokenizer(
            rejected_output,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        base_prompt_length = 0
        for i in range(min(len(chosen_ids), len(rejected_ids))):
            base_prompt_length += 1
            if chosen_ids[i] != rejected_ids[i]:
                break
        last_position = base_prompt_length - 1

        result["chosen_output"] = chosen_ids
        result["rejected_output"] = rejected_ids
        return result, last_position


@dataclass
class ReftRewardCollator:
    tokenizer: transformers.PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []

        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["chosen_output"],
                    "attention_mask": feature["chosen_output_mask"],
                    "reward": (feature["chosen_reward"] if "chosen_reward" in feature else 1.0),
                    "intervention_locations": feature["intervention_locations"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["rejected_output"],
                    "attention_mask": feature["rejected_output_mask"],
                    "reward": (feature["rejected_reward"] if "rejected_reward" in feature else 0.0),
                    "intervention_locations": feature["intervention_locations"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "reward": batch["reward"],
            "intervention_locations": batch["intervention_locations"],
        }
        max_seq_length = batch["input_ids"].shape[-1]
        batch["intervention_locations"] = batch["intervention_locations"][..., :max_seq_length]
        return batch


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


class CommonCollator(object):
    """Collate examples for ReFT."""

    # data_collator: DataCollator
    def __init__(self, data_collator):
        self.data_collator = data_collator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, paddle.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][..., :max_seq_length]
        del batch_inputs["intervention_locations"]
        del batch_inputs["id"]
        print(batch_inputs.keys())
        return batch_inputs
