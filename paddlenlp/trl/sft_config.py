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
from paddlenlp.trainer.trainer_utils import IntervalStrategy
from paddlenlp.trainer.utils.doc import add_start_docstrings
from paddlenlp.transformers.configuration_utils import llmmetaclass


@dataclass
@llmmetaclass
@add_start_docstrings(TrainingArguments.__doc__)
class SFTConfig(TrainingArguments):
    benchmark: bool = field(default=False, metadata={"help": "Whether runs benchmark"})

    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Weather to run benchmark by autotuner. True for from_scratch and pad_max_length."},
    )
    decay_steps: int = field(
        default=0,
        metadata={"help": "The steps use to control the learing rate."},
    )
    tensor_parallel_output: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to output logits in distributed status"},
    )
    unified_checkpoint: bool = field(
        default=True,
        metadata={"help": "Unify hybrid parallel checkpoint."},
    )

    # for data part
    dataset_name_or_path: str = field(default=None, metadata={"help": "Name or path for dataset"})
    task_name: str = field(default=None, metadata={"help": "Additional name to select a more specific task."})
    zero_padding: bool = field(default=False, metadata={"help": "Whether to use Zero Padding data stream"})
    greedy_zero_padding: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Greedy Zero Padding data stream, should be used together with `zero_padding=True`."
        },
    )
    pad_to_multiple_of: int = field(
        default=None, metadata={"help": "If set will pad the sequence to a multiple of the provided value."}
    )
    src_length: int = field(default=1024, metadata={"help": "The maximum length of source(context) tokens."})
    max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum length that model input tokens can have. When Zero Padding is set to True, it's also the maximum length for Zero Padding data stream"
        },
    )
    eval_with_do_generation: bool = field(default=False, metadata={"help": "Whether to do generation for evaluation"})
    save_generation_output: bool = field(
        default=False,
        metadata={"help": "Whether to save generated text to file when eval_with_do_generation set to True."},
    )
    lazy: bool = field(
        default=False,
        metadata={
            "help": "Weather to return `MapDataset` or an `IterDataset`.True for `IterDataset`. False for `MapDataset`."
        },
    )
    chat_template: str = field(
        default=None,
        metadata={
            "help": "the path of `chat_template.json` file to handle multi-rounds conversation. If is None, it will not use `chat_template.json`; If is equal with `model_name_or_path`, it will use the default loading; If is directory, it will find the `chat_template.json` under the directory; If is file, it will load it."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Pad the input sequence to `max_length`."},
    )

    def __post_init__(self):
        super().__post_init__()
        # NOTE(gongenlei): new add autotuner_benchmark
        if self.autotuner_benchmark:
            self.max_steps = 5
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.report_to = []
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO
            self.unified_checkpoint = False
        if self.benchmark:
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.report_to = []
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO
            self.unified_checkpoint = False
