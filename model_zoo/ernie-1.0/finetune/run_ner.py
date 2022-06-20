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

from datasets import load_metric
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import DataCollatorForTokenClassification
from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments,
    Trainer,
)
from paddlenlp.trainer import get_last_checkpoint
from paddlenlp.transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from paddlenlp.utils.log import logger

sys.path.insert(0, os.path.abspath("."))
from token_classification import ner_trans_fn
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

    # set_seed(args)
    data_args.dataset = data_args.dataset.strip()
    if data_args.dataset not in ALL_DATASETS:
        raise ValueError("Not found dataset {}".format(data_args.dataset))

    if data_args.dataset in ALL_DATASETS:
        # if you custom you hyper-parameters in yaml config, it will overwrite all args.
        config = ALL_DATASETS[data_args.dataset]
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

    label_list = getattr(raw_datasets['train'], "label_list", None)
    data_args.label_list = label_list
    data_args.ignore_label = -100
    data_args.no_entity_id = len(data_args.label_list) - 1

    num_classes = 1 if raw_datasets["train"].label_list == None else len(
        raw_datasets['train'].label_list)

    # Define tokenizer, model, loss function.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path, num_classes=num_classes)

    class criterion(nn.Layer):

        def __init__(self):
            super(criterion, self).__init__()
            self.loss_fn = paddle.nn.loss.CrossEntropyLoss(
                ignore_index=data_args.ignore_label)

        def forward(self, *args, **kwargs):
            return paddle.mean(self.loss_fn(*args, **kwargs))

    loss_fct = criterion()

    # Define dataset pre-process function
    trans_fn = partial(ner_trans_fn, tokenizer=tokenizer, args=data_args)
    # Define data collector
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Dataset pre-process
    if training_args.do_train:
        train_dataset = raw_datasets["train"].map(trans_fn)
    if training_args.do_eval:
        eval_dataset = raw_datasets["dev"].map(trans_fn)
    if training_args.do_predict:
        test_dataset = raw_datasets["test"].map(trans_fn)

    # Define the metrics of tasks.
    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [[
            label_list[p] for (p, l) in zip(prediction, label) if l != -100
        ] for prediction, label in zip(predictions, labels)]
        true_labels = [[
            label_list[l] for (p, l) in zip(prediction, label) if l != -100
        ] for prediction, label in zip(predictions, labels)]
        results = metric.compute(predictions=true_predictions,
                                 references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        criterion=loss_fct,
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
