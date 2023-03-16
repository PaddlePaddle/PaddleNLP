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
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
from data import GLUE_1_1_PROCESSED, GLUE_PROCESSED
from utils import GLUE_METRICS, load_pickle, save_pickle

from paddlenlp.data import Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
)
from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer
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
    if "v1_1" in args.cache_dir:
        PROCESSED = GLUE_1_1_PROCESSED
    processed, label = PROCESSED[task_name]
    if label:
        id2label = dict(zip(range(len(label)), label))
    else:
        id2label = None

    is_test = "labels" not in example

    if not is_test:
        if id2label:
            label_text = id2label[example["labels"]]
        else:
            label_text = str(example["labels"])
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
        default="t5-small",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    export_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the exported inference model."},
    )


class T5GlueTrainer(Trainer):
    def __init__(self, do_generation: bool, label2id, **kwargs):
        super().__init__(**kwargs)
        self.do_generation = do_generation
        self.label2id = label2id

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:

        if not self.do_generation:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        all_preds = []
        all_labels = []
        # source_ids, source_mask, labels, target_mask = batch
        labels = inputs["labels"]
        target_mask = inputs["decoder_attention_mask"]

        with paddle.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=5,
            )[0]

        for p, l, m in zip(outputs.numpy(), labels.numpy(), target_mask.numpy()):
            pred = self.tokenizer.decode(p, skip_special_tokens=True).strip()
            label = self.tokenizer.decode(l[m.astype("bool")], skip_special_tokens=True).strip()

            if self.label2id:
                # for classifaction task.
                label = self.label2id[label]
                if pred not in self.label2id:
                    # set to wrong label if the generated text not in the labal set.
                    pred = 0
                    if label == 0:
                        pred = 1
                else:
                    pred = self.label2id[pred]
            else:
                # for regression task.
                label = float(label.replace(" ", ""))
                try:
                    pred = float(pred.replace(" ", ""))
                except Exception:
                    # set to zero if the generated text can not convert to float
                    pred = 0.0

            all_preds.append(pred)
            all_labels.append(label)

        all_preds = paddle.to_tensor(all_preds).detach()
        all_labels = paddle.to_tensor(all_labels).detach()

        return (None, all_preds, all_labels)


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "v1_1" in model_args.model_name_or_path:
        data_args.cache_dir = "./caches_v1_1"
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

    PROCESSED = GLUE_PROCESSED
    if "v1_1" in data_args.cache_dir:
        PROCESSED = GLUE_1_1_PROCESSED
    label_name = PROCESSED[data_args.task_name][1]
    if label_name:
        label2id = dict(zip(label_name, range(len(label_name))))
    else:
        label2id = None
    metric_list = GLUE_METRICS[data_args.task_name]

    # get model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path)

    # get dataloader
    train_dataset = get_train_dataset(tokenizer, data_args)
    if data_args.task_name == "mnli":
        eval_dataset = get_mnli_dev_dataset(tokenizer, data_args, matched=True)
    else:
        eval_dataset = get_dev_dataset(tokenizer, data_args)

    batchify_fn = lambda samples, fn=BatchDict(
        {
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
            "attention_mask": Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # attention_mask
            "labels": Pad(axis=0, pad_val=-100, dtype="int64"),  # lm_labels
            "decoder_attention_mask": Pad(
                axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
            ),  # decoder_attention_mask
        }
    ): fn(samples)
    data_collator = batchify_fn

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        results = {}
        for metric in metric_list:
            results.update(metric(p.label_ids, preds))

        return results

    trainer = T5GlueTrainer(
        model=model,
        criterion=None,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        do_generation=True,
        label2id=label2id,
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


if __name__ == "__main__":
    main()
