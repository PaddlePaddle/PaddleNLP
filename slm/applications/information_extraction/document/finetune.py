# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial
from typing import List, Optional

import paddle
from utils import convert_example, reader

from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    CompressionArguments,
    PdArgumentParser,
    Trainer,
    get_last_checkpoint,
)
from paddlenlp.transformers import UIEX, AutoTokenizer, export_model
from paddlenlp.utils.ie_utils import compute_metrics, uie_loss_func
from paddlenlp.utils.log import logger


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    train_path: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    dev_path: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    max_seq_len: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    dynamic_max_length: Optional[List[int]] = field(
        default=None,
        metadata={"help": "dynamic max length from batch, it can be array of length, eg: 16 32 64 128"},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(default="uie-x-base", metadata={"help": "Path to pretrained model"})
    export_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the exported inference model."},
    )


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ["start_positions", "end_positions"]

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Define model and tokenizer
    model = UIEX.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load and preprocess dataset
    train_ds = load_dataset(reader, data_path=data_args.train_path, max_seq_len=data_args.max_seq_len, lazy=False)
    dev_ds = load_dataset(reader, data_path=data_args.dev_path, max_seq_len=data_args.max_seq_len, lazy=False)
    trans_fn = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=data_args.max_seq_len,
        dynamic_max_length=data_args.dynamic_max_length,
    )
    train_ds = train_ds.map(trans_fn)
    dev_ds = dev_ds.map(trans_fn)

    trainer = Trainer(
        model=model,
        criterion=uie_loss_func,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=dev_ds if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.optimizer = paddle.optimizer.AdamW(
        learning_rate=training_args.learning_rate, parameters=model.parameters()
    )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate and tests model
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)

    # export inference model
    if training_args.do_export:
        # You can also load from certain checkpoint
        # trainer.load_state_dict_from_checkpoint("/path/to/checkpoint/")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
            paddle.static.InputSpec(shape=[None, None, 4], dtype="int64", name="bbox"),
            paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype="int64", name="image"),
        ]
        if model_args.export_model_dir is None:
            model_args.export_model_dir = os.path.join(training_args.output_dir, "export")
        export_model(model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir)


if __name__ == "__main__":
    main()
