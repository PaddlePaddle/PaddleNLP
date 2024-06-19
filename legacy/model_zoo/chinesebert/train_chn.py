# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import paddle
from paddle.metric import Accuracy
from utils import load_ds

from paddlenlp.data import Pad, Stack
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments, set_seed
from paddlenlp.transformers import (
    ChineseBertForSequenceClassification,
    ChineseBertTokenizer,
)


@dataclass
class ModelArguments:
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default="./data",
        metadata={"help": "The path of datasets to be loaded."},
    )


def convert_example(example, tokenizer, max_length=512, is_test=False):
    # The original data is processed into a format that can be read in by the model,
    # enocded_ Inputs is a dict that contains inputs_ids、token_type_ids、etc.
    encoded_inputs = tokenizer(text=example["text"], max_length=max_length)

    # input_ids：After the text is segmented into tokens, the corresponding token id in the vocabulary.
    input_ids = encoded_inputs["input_ids"]
    # # token_type_ids：Does the current token belong to sentence 1 or sentence 2, that is, the segment ids.
    pinyin_ids = encoded_inputs["pinyin_ids"]

    label = np.array([example["label"]], dtype="int64")
    # return encoded_inputs
    return input_ids, pinyin_ids, label


@dataclass
class DataCollator:
    tokenizer: ChineseBertTokenizer

    def __call__(self, features):
        input_ids = []
        pinyin_ids = []
        labels = []
        batch = {}

        for feature in features:
            input_idx, pinyin_idx, label = feature
            input_ids.append(input_idx)
            pinyin_ids.append(pinyin_idx)
            labels.append(label)

        input_ids = (Pad(axis=0, pad_val=self.tokenizer.pad_token_id)(input_ids),)  # input_ids
        pinyin_ids = (Pad(axis=0, pad_val=0)(pinyin_ids),)  # pinyin_ids
        labels = (Stack()(labels),)  # labels

        batch["input_ids"] = input_ids[0]
        batch["pinyin_ids"] = pinyin_ids[0]
        batch["labels"] = labels[0]

        return batch


def compute_metrics(eval_preds):
    labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
    preds = paddle.to_tensor(eval_preds.predictions)
    preds = paddle.nn.functional.softmax(preds, axis=-1)
    labels = paddle.argmax(labels, axis=-1)
    metric = Accuracy()
    correct = metric.compute(preds, labels)
    metric.update(correct)
    acc = metric.accumulate()
    return {"accuracy": acc}


def do_train():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(training_args.seed)

    data_dir = data_args.data_path
    train_path = os.path.join(data_dir, "train.tsv")
    dev_path = os.path.join(data_dir, "dev.tsv")
    test_path = os.path.join(data_dir, "test.tsv")

    train_ds, dev_ds, test_ds = load_ds(datafiles=[train_path, dev_path, test_path])

    model = ChineseBertForSequenceClassification.from_pretrained("ChineseBERT-large", num_classes=2)
    tokenizer = ChineseBertTokenizer.from_pretrained("ChineseBERT-large")

    # Process the data into a data format that the model can read in.
    trans_func = partial(convert_example, tokenizer=tokenizer, max_length=model_args.max_seq_length)
    train_ds = train_ds.map(trans_func, lazy=False)
    dev_ds = dev_ds.map(trans_func, lazy=False)
    test_ds = test_ds.map(trans_func, lazy=False)

    # Form data into batch data, such as padding text sequences of different lengths into the maximum length of batch data,
    # and stack each data label together
    batchify_fn = DataCollator(tokenizer)
    criterion = paddle.nn.loss.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=dev_ds if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=batchify_fn,
        criterion=criterion,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        train_results = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_results.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)


if __name__ == "__main__":
    do_train()
