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

import math
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Optional

import paddle
import paddle.nn as nn
from datasets import load_dataset

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments, set_seed
from paddlenlp.transformers import CodeGenForCausalLM, CodeGenTokenizer
from paddlenlp.utils.log import logger


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="Salesforce/codegen-350M-mono",
        metadata={"help": ("Path to pre-trained model.")},
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to overwrite cache for dataset.")},
    )


@dataclass
class DataArguments:
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input validation data file."},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": ("The training dataset will be truncated in block of this size for training. ")},
    )


def compute_metrics(eval_preds):
    labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
    logits = paddle.to_tensor(eval_preds.predictions)
    loss_fct = nn.CrossEntropyLoss()
    eval_loss = loss_fct(logits[:, :-1, :], labels[:, 1:])
    perplexity = math.exp(eval_loss)
    return {"perplexity": perplexity}


def convert_example(examples, tokenizer):
    """convert examples into necessary features"""
    # Convert raw text to feature
    tokenized_examples = tokenizer(
        examples["code"], return_attention_mask=True, return_position_ids=False, return_token_type_ids=False
    )
    return tokenized_examples


def group_texts(examples, block_size):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def process_ds(dataset, tokenizer, overwrite_cache, block_size):
    trans_func = partial(convert_example, tokenizer=tokenizer)
    dataset = dataset.map(
        trans_func, batched=True, remove_columns=dataset.column_names, load_from_cache_file=overwrite_cache
    )
    trans_func = partial(group_texts, block_size=block_size)
    dataset = dataset.map(trans_func, batched=True, load_from_cache_file=overwrite_cache)
    return dataset


def do_train():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(training_args.seed)

    model = CodeGenForCausalLM.from_pretrained(model_args.model_name_or_path)

    tokenizer = CodeGenTokenizer.from_pretrained(model_args.model_name_or_path)

    train_set = load_dataset("json", data_files=data_args.train_file, split="train")
    dev_set = load_dataset("json", data_files=data_args.validation_file, split="train")

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    train_set = process_ds(train_set, tokenizer, model_args.overwrite_cache, block_size)
    dev_set = process_ds(dev_set, tokenizer, model_args.overwrite_cache, block_size)

    batchify_fn = DataCollatorWithPadding(tokenizer, return_attention_mask=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set if training_args.do_train else None,
        eval_dataset=dev_set if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=batchify_fn,
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
