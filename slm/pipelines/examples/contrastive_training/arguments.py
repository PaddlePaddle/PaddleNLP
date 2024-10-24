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

import os
from dataclasses import dataclass, field
from typing import List, Optional

from paddlenlp.trainer import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )

    normalized: bool = field(default=True)
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})


@dataclass
class DataArguments:
    train_data: str = field(default=None, metadata={"help": "Path to train data"})
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000,
        metadata={"help": "the max number of examples for each dataset"},
    )

    query_instruction_for_retrieval: str = field(default=None, metadata={"help": "instruction for query"})
    passage_instruction_for_retrieval: str = field(default=None, metadata={"help": "instruction for passage"})

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")


@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    margin: Optional[float] = field(default=0.2)
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    sentence_pooling_method: str = field(
        default="mean",
        metadata={"help": "the pooling method, should be weighted_mean"},
    )
    fine_tune_type: str = field(
        default="sft",
        metadata={"help": "fine-tune type for retrieval,eg: sft, bitfit, lora"},
    )
    use_inbatch_neg: bool = field(default=False, metadata={"help": "use passages in the same batch as negatives"})

    use_matryoshka: bool = field(default=False, metadata={"help": "use matryoshka for flexible embedding size"})

    matryoshka_dims: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 768],
        metadata={"help": "matryoshka dims"},
    )
    matryoshka_loss_weights: List[float] = field(
        default_factory=lambda: [1, 1, 1, 1, 1],
        metadata={"help": "matryoshka loss weights"},
    )
