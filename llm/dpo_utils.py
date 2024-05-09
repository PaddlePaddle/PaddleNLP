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

"""DPO utils"""
from dataclasses import dataclass, field

from paddlenlp.trainer import TrainingArguments


def add_start_docstrings(*docstr):
    """Adds docstrings for a function."""

    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class DPOTrainingArguments(TrainingArguments):
    """DPOTrainingArguments"""

    dpo_beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})


@dataclass
class DataArgument:
    """DataArgument"""

    train_dataset_path: str = field(default="./data/train.jsonl", metadata={"help": "Path to the train dataset dir."})
    dev_dataset_path: str = field(default="./data/dev.jsonl", metadata={"help": "Path to the dev dataset dir."})
    max_seq_len: int = field(default=4096, metadata={"help": "Maximum sequence length."})
    max_prompt_len: int = field(default=2048, metadata={"help": "Maximum prompt length."})
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )
    dpo_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )


@dataclass
class ModelArgument:
    """ModelArgument"""

    model_name_or_path: str = field(
        default="ernie-bot-10b", metadata={"help": "Pretrained model name or path to local directory."}
    )
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    recompute_granularity: str = field(
        default="full",
        metadata={
            "help": "The granularity of recompute training can be selected as `full` or `full_attn` or `core_attn`."
        },
    )
    use_attn_mask_start_row_indices: bool = field(
        default=False, metadata={"help": "Whether to use attn_mask_start_row_indices in flash attention."}
    )


def calculate_effective_tokens(training_args, train_dataset, max_seq_len):
    """
    Caculate the effective tokens during training.
    """
    total_effective_tokens = 0
    try:
        data_parallel_degree = training_args.data_parallel_degree
    except:
        data_parallel_degree = 1
    if training_args.sharding_parallel_degree > 1:
        sharding_parallel_degree = training_args.sharding_parallel_degree
    else:
        sharding_parallel_degree = 1

    total_batch = (
        training_args.max_steps
        * training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * sharding_parallel_degree
        * data_parallel_degree
    )
    for i, data in enumerate(train_dataset):
        if i == total_batch:
            break
        total_effective_tokens += len(data["input_ids"])
    total_tokens = total_batch * max_seq_len

    return total_effective_tokens, total_tokens
