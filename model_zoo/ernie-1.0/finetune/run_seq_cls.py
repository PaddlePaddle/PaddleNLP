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
import sys
import yaml
from functools import partial
import distutils.util
import os.path as osp
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.metric import Accuracy
import paddlenlp
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments,
    Trainer,
)
from paddlenlp.trainer import get_last_checkpoint
from paddlenlp.transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from paddlenlp.utils.log import logger

sys.path.insert(0, os.path.abspath("."))
from sequence_classification import seq_trans_fn, clue_trans_fn
from utils import (
    ALL_DATASETS,
    DataArguments,
    ModelArguments,
)


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
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

    data_args.dataset = data_args.dataset.strip()

    if data_args.dataset in ALL_DATASETS:
        # if you custom you hyper-parameters in yaml config, it will overwrite all args.
        config = ALL_DATASETS[data_args.dataset]
        logger.info("Over-writing training config by yaml config!")
        for args in (model_args, data_args, training_args):
            for arg in vars(args):
                if arg in config.keys():
                    setattr(args, arg, config[arg])

        training_args.per_device_train_batch_size = config["batch_size"]
        training_args.per_device_eval_batch_size = config["batch_size"]

    dataset_config = data_args.dataset.split(" ")
    raw_datasets = load_dataset(
        dataset_config[0],
        None if len(dataset_config) <= 1 else dataset_config[1],
    )

    data_args.label_list = getattr(raw_datasets['train'], "label_list", None)
    num_classes = 1 if raw_datasets["train"].label_list == None else len(
        raw_datasets['train'].label_list)

    # Define tokenizer, model, loss function.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_classes=num_classes)
    criterion = nn.loss.CrossEntropyLoss(
    ) if data_args.label_list else nn.loss.MSELoss()

    # Define dataset pre-process function
    if "clue" in data_args.dataset:
        trans_fn = partial(clue_trans_fn, tokenizer=tokenizer, args=data_args)
    else:
        trans_fn = partial(seq_trans_fn, tokenizer=tokenizer, args=data_args)

    # Define data collector
    data_collator = DataCollatorWithPadding(tokenizer)

    # Dataset pre-process
    if training_args.do_train:
        train_dataset = raw_datasets["train"].map(trans_fn)
    if training_args.do_eval:
        eval_dataset = raw_datasets["dev"].map(trans_fn)
    if training_args.do_predict:
        test_dataset = raw_datasets["test"].map(trans_fn)

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions

        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        probs = F.softmax(preds, axis=1)
        metric = Accuracy()
        metric.reset()
        result = metric.compute(preds, label)
        metric.update(result)
        accu = metric.accumulate()
        metric.reset()
        return {"accuracy": accu}

    trainer = Trainer(
        model=model,
        criterion=criterion,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
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

    if training_args.do_predict:
        test_ret = trainer.predict(test_dataset)
        trainer.log_metrics("test", test_ret.metrics)
        if test_ret.label_ids is None:
            paddle.save(
                test_ret.predictions,
                os.path.join(training_args.output_dir, "test_results.pdtensor"),
            )

    # export inference model
    if training_args.do_export:
        # You can also load from certain checkpoint
        # trainer.load_state_dict_from_checkpoint("/path/to/checkpoint/")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64")  # segment_ids
        ]
        if model_args.export_model_dir is None:
            model_args.export_model_dir = os.path.join(training_args.output_dir,
                                                       "export")
        paddlenlp.transformers.export_model(model=trainer.model,
                                            input_spec=input_spec,
                                            path=model_args.export_model_dir)


if __name__ == "__main__":
    main()
