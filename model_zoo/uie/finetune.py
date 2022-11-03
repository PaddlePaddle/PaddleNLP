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

import argparse
import time
import os
from functools import partial
from typing import Optional
from dataclasses import dataclass, field

import paddle
from paddle.utils.download import get_path_from_url
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer, export_model
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, CompressionArguments, Trainer
from paddlenlp.trainer import get_last_checkpoint
from paddlenlp.utils.log import logger

from model import UIE, UIEM
from utils import reader, MODEL_MAP, map_offset, convert_example


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to 
    specify them on the command line.
    """
    train_path: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        })

    dev_path: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        })

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default="uie-base",
        metadata={
            "help":
            "Path to pretrained model, such as 'uie-base', 'uie-tiny', " \
            "'uie-medium', 'uie-mini', 'uie-micro', 'uie-nano', 'uie-base-en', " \
            "'uie-m-base', 'uie-m-large', or finetuned model path."
        })
    export_model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory to store the exported inference model."
        },
    )
    multilingual: bool = field(
        default=False,
        metadata={"help": "Whether the model is a multilingual model."})


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir
    ) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(
                training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    if model_args.model_name_or_path in MODEL_MAP:
        resource_file_urls = MODEL_MAP[
            model_args.model_name_or_path]['resource_file_urls']

        logger.info("Downloading resource files...")
        for key, val in resource_file_urls.items():
            file_path = os.path.join(model_args.model_name_or_path, key)
            if not os.path.exists(file_path):
                get_path_from_url(val, model_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if model_args.multilingual:
        model = UIEM.from_pretrained(model_args.model_name_or_path)
    else:
        model = UIE.from_pretrained(model_args.model_name_or_path)

    train_ds = load_dataset(reader,
                            data_path=data_args.train_path,
                            max_seq_len=data_args.max_seq_length,
                            lazy=False)
    dev_ds = load_dataset(reader,
                          data_path=data_args.dev_path,
                          max_seq_len=data_args.max_seq_length,
                          lazy=False)

    trans_fn = partial(convert_example,
                       tokenizer=tokenizer,
                       max_seq_len=data_args.max_seq_length,
                       multilingual=model_args.multilingual)

    train_ds = train_ds.map(trans_fn)
    dev_ds = dev_ds.map(trans_fn)

    data_collator = DataCollatorWithPadding(tokenizer)

    criterion = paddle.nn.BCELoss()

    def uie_loss_func(outputs, labels):
        start_ids, end_ids = labels
        start_prob, end_prob = outputs
        start_ids = paddle.cast(start_ids, 'float32')
        end_ids = paddle.cast(end_ids, 'float32')
        loss_start = criterion(start_prob, start_ids)
        loss_end = criterion(end_prob, end_ids)
        loss = (loss_start + loss_end) / 2.0
        return loss

    def compute_metrics(p):
        metric = SpanEvaluator()
        start_prob, end_prob = p.predictions
        start_ids, end_ids = p.label_ids
        metric.reset()

        num_correct, num_infer, num_label = metric.compute(
            start_prob, end_prob, start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)
        precision, recall, f1 = metric.accumulate()
        metric.reset()

        return {"precision": precision, "recall": recall, "f1": f1}

    trainer = Trainer(
        model=model,
        criterion=uie_loss_func,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds
        if training_args.do_train or training_args.do_compress else None,
        eval_dataset=dev_ds
        if training_args.do_eval or training_args.do_compress else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.optimizer = paddle.optimizer.AdamW(
        learning_rate=training_args.learning_rate,
        parameters=model.parameters())
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
        if model_args.multilingual:
            input_spec = [
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='input_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='pos_ids'),
            ]
        else:
            input_spec = [
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='input_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='token_type_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='pos_ids'),
                paddle.static.InputSpec(shape=[None, None],
                                        dtype="int64",
                                        name='att_mask'),
            ]
        if model_args.export_model_dir is None:
            model_args.export_model_dir = os.path.join(training_args.output_dir,
                                                       "export")
        export_model(model=trainer.model,
                     input_spec=input_spec,
                     path=model_args.export_model_dir)
    if training_args.do_compress:

        @paddle.no_grad()
        def custom_evaluate(self, model, data_loader):
            metric = SpanEvaluator()
            model.eval()
            metric.reset()
            for batch in data_loader:
                if model_args.multilingual:
                    logits = model(input_ids=batch['input_ids'],
                                   pos_ids=batch["pos_ids"])
                else:
                    logits = model(input_ids=batch['input_ids'],
                                   token_type_ids=batch['token_type_ids'],
                                   pos_ids=batch["pos_ids"],
                                   att_mask=batch["att_mask"])
                start_prob, end_prob = logits
                start_ids, end_ids = batch["start_positions"], batch[
                    "end_positions"]
                num_correct, num_infer, num_label = metric.compute(
                    start_prob, end_prob, start_ids, end_ids)
                metric.update(num_correct, num_infer, num_label)
            precision, recall, f1 = metric.accumulate()
            logger.info("f1: %s, precision: %s, recall: %s" %
                        (f1, precision, f1))
            model.train()
            return f1

        trainer.compress(custom_evaluate=custom_evaluate)


if __name__ == "__main__":
    main()
