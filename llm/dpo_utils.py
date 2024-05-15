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
    loss_type: str = field(default="sigmoid", metadata={"help": "the loss type for DPO"})


@dataclass
class DataArgument:
    """DataArgument"""

    dataset_name_or_path: str = field(default="./data/", metadata={"help": "Path to the dataset dir."})
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum sequence length."})
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
