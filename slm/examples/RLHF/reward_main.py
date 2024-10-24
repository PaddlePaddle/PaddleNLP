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
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import paddle

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from data import PreferenceDataset, parse_dataset
from models import AutoModelForScore
from reward_trainer import RewardTrainer

from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
from paddlenlp.utils.log import logger


@dataclass
class TrainingArguments(TrainingArguments):
    loss_type: Literal["token-wise", "sequence-wise"] = field(
        default="sequence-wise",
        metadata={
            "help": "Calculate ranking loss with all token-wise reward outputs in the sequence or the "
            "sequence-wise reward output only (the reward of the last token in each sequence)."
        },
    )
    # regularization
    regularization: float = field(
        default=0.0,
        metadata={"help": "The regularization strength for the L2 regularization for score outputs."},
    )


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    normalize_score_during_training: bool = field(
        default=False, metadata={"help": "Whether to normalize score during training."}
    )
    normalizer_type: Literal["RunningMeanStd", "ExponentialMovingAverage"] = field(
        default=None, metadata={"help": "The type of the reward normalizer."}
    )
    normalizer_momentum: float = field(
        default=None,
        metadata={
            "help": "The momentum use in ExponentialMovingAverage, EMA_{t+1} = momentum * x + (1 - momentum) * EMA_t."
        },
    )
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})

    @property
    def extra_model_kwargs(self):
        """Extra keyword arguments for initializing the model."""
        return {
            "score_type": "reward",
            "do_normalize": self.normalize_score_during_training,
            "normalizer_type": self.normalizer_type,
            "momentum": self.normalizer_momentum,
        }


@dataclass
class DataArgument:
    dataset_name_or_path: str = field(default=None, metadata={"help": "Name or path for dataset"})
    task_name: str = field(default=None, metadata={"help": "Additional name to select a more specific task."})
    train_datasets: str = field(default=None, metadata={"help": "Dataset name(s) registered in the raw dataset."})
    eval_datasets: str = field(default=None, metadata={"help": "Dataset name(s) registered in the raw dataset."})
    max_length: int = field(
        default=2048,
        metadata={"help": "The maximum length that model input tokens can ."},
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

    @property
    def parsed_train_datasets(self) -> Tuple[str, Dict[str, Any]]:
        """Parse dataset path and its proportion and optionally additional arguments from `train_datasets`."""
        return [parse_dataset(string) for string in self.train_datasets.split(",")]

    @property
    def parsed_eval_datasets(self) -> Tuple[str, Dict[str, Any]]:
        """Parse dataset path and its proportion and optionally additional arguments from `eval_datasets`."""
        return [parse_dataset(string) for string in self.eval_datasets.split(",")]


def main():
    # Arguments
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load model
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16:
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"

    if training_args.pipeline_parallel_degree > 1:
        raise ValueError("Not support pipeline parallel mode.")
    else:
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            tensor_parallel_output=False,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
            dtype=dtype,
        )
        if hasattr(model_config, "use_flash_attention"):
            model_config.use_flash_attention = model_args.use_flash_attention
        model = AutoModelForScore.from_pretrained(
            model_args.model_name_or_path, config=model_config, **model_args.extra_model_kwargs
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=data_args.max_length, padding_side="right"
    )
    if isinstance(tokenizer, LlamaTokenizer) and tokenizer.pad_token_id is None:
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        # to be consistent with PKU-Alignment/alpaca-7b-reproduced
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds = PreferenceDataset(data_args.parsed_train_datasets, tokenizer=tokenizer)

    dev_ds = PreferenceDataset(data_args.parsed_eval_datasets, tokenizer=tokenizer)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=train_ds.get_collator(),
    )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
