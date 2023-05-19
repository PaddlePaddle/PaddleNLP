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

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import paddle
from datasets import load_dataset
from utils import PegasusTrainer, compute_metrics, convert_example, main_process_first

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, set_seed
from paddlenlp.transformers import (
    PegasusChineseTokenizer,
    PegasusForConditionalGeneration,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese",
        metadata={"help": ("Path to pre-trained model.")},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after "
                "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    min_target_length: Optional[int] = field(
        default=0,
        metadata={"help": ("The minimum total sequence length for target text when generating. ")},
    )
    max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "The maximum total sequence length for target text after "
                "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    use_SSTIA: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to use SSTIA.")},
    )
    mix_ratio: Optional[float] = field(
        default=0,
        metadata={"help": ("Mixture ratio for TSDASG synthetic input.")},
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": ("The number of beams to use in beam search.")},
    )
    predict_with_generate: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to generate in predcit.")},
    )


@dataclass
class DataArguments:
    train_file: Optional[str] = field(
        default="data/train.json",
        metadata={"help": ("Train data path.")},
    )
    eval_file: Optional[str] = field(
        default="data/test.json",
        metadata={"help": ("Eval data path.")},
    )


def compute_metrics_trainer(eval_preds, tokenizer):
    all_preds = []
    all_labels = []
    labels = eval_preds.label_ids
    preds = eval_preds.predictions
    all_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    all_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    rougel = compute_metrics(all_preds, all_labels)
    return {"RougeL": rougel}


def do_train():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(training_args.seed)

    training_args.generation_max_length = model_args.max_target_length
    training_args.generation_num_beams = model_args.num_beams
    training_args.predict_with_generate = model_args.predict_with_generate

    tokenizer = PegasusChineseTokenizer.from_pretrained(model_args.model_name_or_path)
    train_set = load_dataset("json", data_files=data_args.train_file, split="train")
    dev_set = load_dataset("json", data_files=data_args.eval_file, split="train")
    remove_columns = ["content", "title"]
    trans_func = partial(
        convert_example,
        text_column="content",
        summary_column="title",
        tokenizer=tokenizer,
        max_source_length=model_args.max_source_length,
        max_target_length=model_args.max_target_length,
    )
    with main_process_first(desc="train dataset map pre-processing"):
        train_set = train_set.map(trans_func, batched=True, load_from_cache_file=True, remove_columns=remove_columns)
    with main_process_first(desc="dev dataset map pre-processing"):
        dev_set = dev_set.map(trans_func, batched=True, load_from_cache_file=True, remove_columns=remove_columns)

    model = PegasusForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    if model_args.use_SSTIA:
        model.use_SSTIA = True
        model.mix_ratio = model_args.mix_ratio

    batchify_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    compute_metrics_func = partial(
        compute_metrics_trainer,
        tokenizer=tokenizer,
    )

    trainer = PegasusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set if training_args.do_train else None,
        eval_dataset=dev_set if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=batchify_fn,
        compute_metrics=compute_metrics_func,
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
