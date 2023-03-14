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
from typing import Optional

import paddle
from data import GLUE_PROCESSED
from modeling import LLaMAForCausalLM
from tokenizer import LLaMATokenizer
from utils import GLUE_METRICS, load_pickle, save_pickle

from paddlenlp.data import DefaultDataCollator
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
)
from paddlenlp.utils.log import logger

label_length_map = {
    "cola": 4,
    "sst-2": 1,
    "mrpc": 5,
    "sts-b": 5,
    "qqp": 5,
    "mnli": 4,
    "qnli": 5,
    "rte": 5,
}


def trans_func(example, tokenizer, args):
    task_name = args.task_name
    PROCESSED = GLUE_PROCESSED
    processed, label = PROCESSED[task_name]
    if label:
        id2label = dict(zip(range(len(label)), label))
    else:
        id2label = None

    is_test = "label" not in example

    if not is_test:
        if id2label:
            label_text = id2label[example["label"]]
        else:
            label_text = str(example["label"])
        target = tokenizer(label_text, return_token_type_ids=False, return_attention_mask=True)

    if len(processed) == 1:
        text = processed[0] + example["sentence"]
    else:
        text = processed[0] + example["sentence1"] + processed[1] + example["sentence2"]

    source = tokenizer(
        text,
        max_seq_len=args.max_seq_length,
        padding="max_length",
        return_token_type_ids=False,
        return_attention_mask=True,
    )

    if not is_test:
        return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": target["input_ids"],
            "decoder_attention_mask": target["attention_mask"],
        }
    else:
        return {"input_ids": source["input_ids"], "attention_mask": source["attention_mask"]}


class BatchDict(object):
    def __init__(self, fn):
        assert isinstance(fn, (dict)), (
            "Input pattern not understood. The input of Dict must be a dict with key of input column name and value of collate_fn "
            "Received fn=%s" % (str(fn))
        )

        self._fn = fn

        for col_name, ele_fn in self._fn.items():
            assert callable(ele_fn), "Batchify functions must be callable! type(fn[%d]) = %s" % (
                col_name,
                str(type(ele_fn)),
            )

    def __call__(self, data):

        ret = {}
        if len(data) <= 0:
            return ret

        for col_name, ele_fn in self._fn.items():
            # skip unused col_name, such as labels in test mode.
            if col_name not in data[0].keys():
                continue
            result = ele_fn([ele[col_name] for ele in data])
            ret[col_name] = result

        return ret


def get_train_dataset(tokenizer, args):
    filename = os.path.join(args.cache_dir, args.task_name + "_train" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits="train")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)
    return ds


def get_dev_dataset(tokenizer, args):
    filename = os.path.join(args.cache_dir, args.task_name + "_dev" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits="dev")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    return ds


def get_mnli_dev_dataset(tokenizer, args, matched=True):
    if matched:
        split = "dev_matched"
    else:
        split = "dev_mismatched"
    filename = os.path.join(args.cache_dir, args.task_name + f"_{split}" + ".pkl")
    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits=split)
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    return ds


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(default=None, metadata={"help": "The name of the task to use (via the datasets library)."})

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    cache_dir: str = field(default="./caches", metadata={"help": "cache dir for datasets."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="llama-7b",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    export_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the exported inference model."},
    )


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not os.path.exists(data_args.cache_dir):
        os.mkdir(data_args.cache_dir)

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
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

    metric_list = GLUE_METRICS[data_args.task_name]

    # get model and tokenizer
    model = LLaMAForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = LLaMATokenizer.from_pretrained(model_args.model_name_or_path)

    # get dataloader
    train_dataset = get_train_dataset(tokenizer, data_args)
    eval_dataset = get_dev_dataset(tokenizer, data_args)

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        results = {}
        for metric in metric_list:
            results.update(metric(p.label_ids, preds))

        return results

    data_collator = DefaultDataCollator()

    training_args.predict_with_generate = True
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
        # trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate and tests model
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
