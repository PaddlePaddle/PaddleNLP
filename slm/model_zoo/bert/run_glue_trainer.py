# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
from datasets import load_dataset
from paddle.metric import Accuracy

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    ErnieForSequenceClassification,
    ErnieTokenizer,
)

METRIC_CLASSES = {
    "cola": Mcc,
    "sst2": Accuracy,
    "mrpc": AccuracyAndF1,
    "stsb": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
    "wnli": Accuracy,
}

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "ernie": (ErnieForSequenceClassification, ErnieTokenizer),
}


@dataclass
class ModelArguments:
    task_name: str = field(
        default=None,
        metadata={"help": "The name of the task to train selected in the list: " + ", ".join(METRIC_CLASSES.keys())},
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pre-trained model or shortcut name"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


def do_train():
    training_args, model_args = PdArgumentParser([TrainingArguments, ModelArguments]).parse_args_into_dataclasses()
    training_args: TrainingArguments = training_args
    model_args: ModelArguments = model_args

    training_args.print_config(model_args, "Model")
    training_args.print_config(training_args, "Training")

    model_args.task_name = model_args.task_name.lower()

    sentence1_key, sentence2_key = task_to_keys[model_args.task_name]

    train_ds = load_dataset("glue", model_args.task_name, split="train", trust_remote_code=True)
    columns = train_ds.column_names
    is_regression = model_args.task_name == "stsb"
    label_list = None
    if not is_regression:
        label_list = train_ds.features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, max_length=model_args.max_seq_length, truncation=True)
        if "label" in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=columns)
    data_collator = DataCollatorWithPadding(tokenizer)

    if model_args.task_name == "mnli":
        dev_ds_matched, dev_ds_mismatched = load_dataset(
            "glue", model_args.task_name, split=["validation_matched", "validation_mismatched"]
        )
        dev_ds_matched = dev_ds_matched.map(preprocess_function, batched=True, remove_columns=columns)
        dev_ds_mismatched = dev_ds_mismatched.map(preprocess_function, batched=True, remove_columns=columns)
        dev_ds = {"matched": dev_ds_matched, "mismatched": dev_ds_mismatched}
    else:
        dev_ds = load_dataset("glue", model_args.task_name, split="validation")
        dev_ds = dev_ds.map(preprocess_function, batched=True, remove_columns=columns)

    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        metric = METRIC_CLASSES[model_args.task_name]()
        result = metric.compute(preds, label)
        metric.update(result)

        if isinstance(metric, AccuracyAndF1):
            acc, precision, recall, f1, _ = metric.accumulate()
            return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
        elif isinstance(metric, Mcc):
            mcc = metric.accumulate()
            return {"mcc": mcc[0]}
        elif isinstance(metric, PearsonAndSpearman):
            pearson, spearman, _ = metric.accumulate()
            return {"pearson": pearson, "spearman": spearman}
        elif isinstance(metric, Accuracy):
            acc = metric.accumulate()
            return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        if model_args.task_name == "mnli":
            for _, eval_dataset in dev_ds.items():
                eval_metrics = trainer.evaluate(eval_dataset)
                trainer.log_metrics("eval", eval_metrics)
                trainer.save_metrics("eval", eval_metrics)
        else:
            eval_metrics = trainer.evaluate(dev_ds)
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    do_train()
