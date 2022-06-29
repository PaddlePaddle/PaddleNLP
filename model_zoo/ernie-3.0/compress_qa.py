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
import paddlenlp
from paddlenlp.data import DataCollatorWithPadding

from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments,
    Trainer,
)
from paddlenlp.trainer import EvalPrediction, get_last_checkpoint
from paddlenlp.transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
from compress_trainer import CompressConfig, PTQConfig
from paddlenlp.utils.log import logger
from datasets import load_metric, load_dataset

sys.path.append("../ernie-1.0/finetune")
from question_answering import (
    QuestionAnsweringTrainer,
    CrossEntropyLossForSQuAD,
    prepare_train_features,
    prepare_validation_features,
)
from utils import (
    ALL_DATASETS,
    DataArguments,
    ModelArguments,
)


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    data_args.dataset = data_args.dataset.strip()

    if data_args.dataset in ALL_DATASETS:
        # if you custom you hyper-parameters in yaml config, it will overwrite all args.
        config = ALL_DATASETS[data_args.dataset]
        for args in (model_args, data_args, training_args):
            for arg in vars(args):
                if arg in config.keys():
                    setattr(args, arg, config[arg])

        training_args.per_device_train_batch_size = config["batch_size"]
        training_args.per_device_eval_batch_size = config["batch_size"]

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    dataset_config = data_args.dataset.split(" ")
    raw_datasets = load_dataset(
        dataset_config[0],
        None if len(dataset_config) <= 1 else dataset_config[1],
        cache_dir=model_args.cache_dir)

    label_list = getattr(raw_datasets['train'], "label_list", None)
    data_args.label_list = label_list

    # Define tokenizer, model, loss function.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path)

    loss_fct = CrossEntropyLossForSQuAD()

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    column_names = raw_datasets["train"].column_names

    column_names = raw_datasets["validation"].column_names

    train_dataset = raw_datasets["train"]
    # Create train feature from dataset
    with training_args.main_process_first(
            desc="train dataset map pre-processing"):
        # Dataset pre-process
        train_dataset = train_dataset.map(
            partial(prepare_train_features, tokenizer=tokenizer,
                    args=data_args),
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
    eval_examples = raw_datasets["validation"]
    with training_args.main_process_first(
            desc="evaluate dataset map pre-processing"):
        eval_dataset = eval_examples.map(
            partial(prepare_validation_features,
                    tokenizer=tokenizer,
                    args=data_args),
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    # Define data collector
    data_collator = DataCollatorWithPadding(tokenizer)

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions, all_nbest_json, scores_diff_json = compute_prediction(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
        )

        references = [{
            "id": ex["id"],
            "answers": ex["answers"]
        } for ex in examples]
        return EvalPrediction(predictions=predictions, label_ids=references)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        tokenizer=tokenizer)

    output_dir = os.path.join(model_args.model_name_or_path, "compress")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prune = True
    compress_config = CompressConfig(quantization_config=PTQConfig(
        algo_list=['hist', 'mse'], batch_size_list=[4, 8, 16]))
    trainer.compress(output_dir,
                     pruning=prune,
                     quantization=True,
                     compress_config=compress_config)


if __name__ == "__main__":
    main()
