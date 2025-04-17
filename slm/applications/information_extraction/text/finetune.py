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

import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import paddle
from utils import convert_example, reader

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.trainer import (
    CompressionArguments,
    PdArgumentParser,
    Trainer,
    get_last_checkpoint,
)
from paddlenlp.transformers import UIE, UIEM, AutoTokenizer, export_model
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

    max_seq_length: Optional[int] = field(
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

    model_name_or_path: Optional[str] = field(
        default="uie-base",
        metadata={
            "help": "Path to pretrained model, such as 'uie-base', 'uie-tiny', "
            "'uie-medium', 'uie-mini', 'uie-micro', 'uie-nano', 'uie-base-en', "
            "'uie-m-base', 'uie-m-large', or finetuned model path."
        },
    )
    export_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the exported inference model."},
    )
    multilingual: bool = field(default=False, metadata={"help": "Whether the model is a multilingual model."})


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ["start_positions", "end_positions"]

    if model_args.model_name_or_path in ["uie-m-base", "uie-m-large"]:
        model_args.multilingual = True
    elif os.path.exists(os.path.join(model_args.model_name_or_path, "model_config.json")):
        with open(os.path.join(model_args.model_name_or_path, "model_config.json")) as f:
            init_class = json.load(f)["init_class"]
        if init_class == "UIEM":
            model_args.multilingual = True

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

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if model_args.multilingual:
        model = UIEM.from_pretrained(model_args.model_name_or_path)
    else:
        model = UIE.from_pretrained(model_args.model_name_or_path)

    train_ds = load_dataset(reader, data_path=data_args.train_path, max_seq_len=data_args.max_seq_length, lazy=False)
    dev_ds = load_dataset(reader, data_path=data_args.dev_path, max_seq_len=data_args.max_seq_length, lazy=False)

    trans_fn = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=data_args.max_seq_length,
        multilingual=model_args.multilingual,
        dynamic_max_length=data_args.dynamic_max_length,
    )

    train_ds = train_ds.map(trans_fn)
    dev_ds = dev_ds.map(trans_fn)

    if training_args.device == "npu":
        data_collator = DataCollatorWithPadding(tokenizer, padding="longest")
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        criterion=uie_loss_func,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds if training_args.do_train or training_args.do_compress else None,
        eval_dataset=dev_ds if training_args.do_eval or training_args.do_compress else None,
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
        if training_args.device == "npu":
            # npu will transform int64 to int32 for internal calculation.
            # To reduce useless transformation, we feed int32 inputs.
            input_spec_dtype = "int32"
        else:
            input_spec_dtype = "int64"
        if model_args.multilingual:
            input_spec = [
                paddle.static.InputSpec(shape=[None, None], dtype=input_spec_dtype, name="input_ids"),
                paddle.static.InputSpec(shape=[None, None], dtype=input_spec_dtype, name="position_ids"),
            ]
        else:
            input_spec = [
                paddle.static.InputSpec(shape=[None, None], dtype=input_spec_dtype, name="input_ids"),
                paddle.static.InputSpec(shape=[None, None], dtype=input_spec_dtype, name="token_type_ids"),
                paddle.static.InputSpec(shape=[None, None], dtype=input_spec_dtype, name="position_ids"),
                paddle.static.InputSpec(shape=[None, None], dtype=input_spec_dtype, name="attention_mask"),
            ]
        if model_args.export_model_dir is None:
            model_args.export_model_dir = os.path.join(training_args.output_dir, "export")
        export_model(model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir)
    if training_args.do_compress:

        @paddle.no_grad()
        def custom_evaluate(self, model, data_loader):
            metric = SpanEvaluator()
            model.eval()
            metric.reset()
            for batch in data_loader:
                if model_args.multilingual:
                    logits = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"])
                else:
                    logits = model(
                        input_ids=batch["input_ids"],
                        token_type_ids=batch["token_type_ids"],
                        position_ids=batch["position_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                start_prob, end_prob = logits
                start_ids, end_ids = batch["start_positions"], batch["end_positions"]
                num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
                metric.update(num_correct, num_infer, num_label)
            precision, recall, f1 = metric.accumulate()
            logger.info("f1: %s, precision: %s, recall: %s" % (f1, precision, f1))
            model.train()
            return f1

        trainer.compress(custom_evaluate=custom_evaluate)


if __name__ == "__main__":
    main()
