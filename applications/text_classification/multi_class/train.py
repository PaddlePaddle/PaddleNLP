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

import functools
import os
from dataclasses import dataclass, field

import paddle
from paddle.metric import Accuracy
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import PdArgumentParser, EarlyStoppingCallback, TrainingArguments, Trainer
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import preprocess_function, read_local_dataset

SUPPORTED_MODELS = [
    "ernie-1.0-large-zh-cw",
    "ernie-3.0-xbase-zh",
    "ernie-3.0-base-zh",
    "ernie-3.0-medium-zh",
    "ernie-3.0-micro-zh",
    "ernie-3.0-mini-zh",
    "ernie-3.0-nano-zh",
    "ernie-2.0-base-en",
    "ernie-2.0-large-en",
    "ernie-m-base",
    "ernie-m-large",
]


# yapf: disable
@dataclass
class DataArguments:
    data_dir: str = field(default="./data/", metadata={"help": "Path to a dataset which includes train.txt, dev.txt, test.txt, label.txt and data.txt (optional)."})
    max_seq_length: int = field(default=128, metadata={"help": "Maximum number of tokens for the model"})
    early_stopping_patience: int = field(default=4, metadata={"help": "Stop training when the specified metric worsens for early_stopping_patience evaluation calls"})

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-3.0-base-zh", metadata={"help": "Build-in pretrained model name or the path to local model."})
# yapf: enable


def main():
    """
    Training a binary or multi classification model
    """

    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    print(not os.path.isdir(model_args.model_name_or_path))
    print(model_args.model_name_or_path not in SUPPORTED_MODELS)
    if not os.path.isdir(model_args.model_name_or_path) and model_args.model_name_or_path not in SUPPORTED_MODELS:
        raise ValueError(
            f"{model_args.model_name_or_path} is not a supported model type. Either use a local model path or select a model from {SUPPORTED_MODELS}"
        )

    paddle.set_device(training_args.device)

    # load and preprocess dataset
    label_list = {}
    with open(os.path.join(data_args.data_dir, "label.txt"), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.strip()
            label_list[l] = i

    train_ds = load_dataset(
        read_local_dataset, path=os.path.join(data_args.data_dir, "train.txt"), label_list=label_list, lazy=False
    )
    dev_ds = load_dataset(
        read_local_dataset, path=os.path.join(data_args.data_dir, "dev.txt"), label_list=label_list, lazy=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=data_args.max_seq_length)
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)

    # Define model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_classes=len(label_list)
    )

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = Accuracy()
        correct = metric.compute(paddle.to_tensor(eval_preds.predictions), paddle.to_tensor(eval_preds.label_ids))
        metric.update(correct)
        acc = metric.accumulate()
        return {"accuracy": acc}

    # Define the early-stopping callback.
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience, early_stopping_threshold=0.0)
    ]

    # Define loss function
    criterion = paddle.nn.loss.CrossEntropyLoss()

    # Define Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=callbacks,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
