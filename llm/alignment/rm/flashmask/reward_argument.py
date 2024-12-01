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


from dataclasses import dataclass, field
from typing import Optional

from paddlenlp.trainer import TrainingArguments
from paddlenlp.trainer.utils.doc import add_start_docstrings


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArguments(TrainingArguments):
    """TrainingArguments"""

    unified_checkpoint: bool = field(
        default=True,
        metadata={"help": "Enable fused linear grad add strategy."},
    )

    unified_checkpoint_config: Optional[str] = field(
        default="",
        metadata={"help": "Configs to unify hybrid parallel checkpoint.\n"},
    )


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
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )
    greedy_zero_padding: bool = field(
        default=False,
        metadata={"help": "Whether to use Greedy Zero Padding data stream."},
    )
    lazy: bool = field(
        default=False,
        metadata={
            "help": "Weather to return `MapDataset` or an `IterDataset`.True for `IterDataset`. False for `MapDataset`."
        },
    )


@dataclass
class ModelArgument:
    """ModelArgument"""

    model_name_or_path: str = field(
        default=None, metadata={"help": "Pretrained model name or path to local directory."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    recompute_granularity: str = field(
        default="full",
        metadata={
            "help": "The granularity of recompute training can be selected as `full` or `full_attn` or `core_attn`."
        },
    )
    flash_mask: bool = field(default=False, metadata={"help": "Whether to use flash mask in flash attention."})
    virtual_pp_degree: int = field(
        default=1,
        metadata={"help": "virtual_pp_degree"},
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={"help": "whether to use sequence parallel"},
    )
