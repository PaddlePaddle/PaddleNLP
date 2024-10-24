# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
# flake8: noqa
import os
from dataclasses import dataclass, field

import paddle
from data_util import get_train_eval_dataset
from paddle.metric import Accuracy
from trainer_util import ErnieViLTrainer

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.transformers import ErnieViLModel, ErnieViLTokenizer

os.environ["NCCL_DEBUG"] = "INFO"


@dataclass
class DataArguments:
    test_only: bool = field(default=False, metadata={"help": "Whether to evaluate model on public test datasets."})
    data_root: str = field(
        default="./data",
        metadata={"help": "Whether to evaluate model on public test datasets."},
    )
    train_data: str = field(
        default="./data",
        metadata={"help": "Whether to evaluate model on public test datasets."},
    )
    val_data: str = field(
        default="./data",
        metadata={"help": "Whether to evaluate model on public test datasets."},
    )
    max_seq_len: int = field(
        default=32,
        metadata={"help": "The length of the text."},
    )


@dataclass
class ModelArguments:
    checkpoint_path: str = field(default="", metadata={"help": "checkpoint path"})
    model_type: str = field(default="ernie_vil-2.0-base-zh", metadata={"help": "the type of model"})
    model_name_or_path: str = field(
        default="PaddlePaddle/ernie_vil-2.0-base-zh", metadata={"help": "model name or path for initialization"}
    )


def do_train():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)
    if paddle.distributed.is_initialized() and paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    tokenizer = ErnieViLTokenizer.from_pretrained(model_args.model_name_or_path)
    train_dataset, eval_dataset = get_train_eval_dataset(
        data_args, tokenizer=tokenizer, max_txt_length=data_args.max_seq_len
    )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    my_collate = DataCollatorWithPadding(tokenizer)
    model = ErnieViLModel.from_pretrained(model_args.model_name_or_path)

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        metric = Accuracy()
        result = metric.compute(preds, label)
        metric.update(result)
        accu = metric.accumulate()
        return {"accuracy": accu}

    trainer = ErnieViLTrainer(
        model=model,
        criterion=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=my_collate,
        callbacks=None,
        compute_metrics=compute_metrics,
    )
    if data_args.test_only:
        train_result = trainer.evaluate()
    elif training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    print(train_result)


if __name__ == "__main__":
    do_train()
