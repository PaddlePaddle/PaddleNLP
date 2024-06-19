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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import paddle
from cmrc_evaluate import get_result
from dataset_cmrc2018 import EvalTrainer, get_dev_dataset, get_train_dataset
from metric_cmrc import compute_prediction, squad_evaluate
from utils import CrossEntropyLossForSQuAD, save_json

from paddlenlp.data import Pad, Stack
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, set_seed
from paddlenlp.transformers import (
    BertForQuestionAnswering,
    BertTokenizer,
    ChineseBertForQuestionAnswering,
    ChineseBertTokenizer,
    ErnieForQuestionAnswering,
    ErnieTokenizer,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "ernie": (ErnieForQuestionAnswering, ErnieTokenizer),
    "chinesebert": (ChineseBertForQuestionAnswering, ChineseBertTokenizer),
}


@dataclass
class ModelArguments:
    model_type: Optional[str] = field(
        default="chinesebert",
        metadata={"help": ("Type of pre-trained model.")},
    )
    model_name_or_path: Optional[str] = field(
        default="ChineseBERT-large",
        metadata={"help": ("Path to pre-trained model or shortcut name of model.")},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    doc_stride: Optional[int] = field(
        default=128,
        metadata={"help": ("When splitting up a long document into chunks, how much stride to take between chunks.")},
    )
    n_best_size: Optional[int] = field(
        default=35,
        metadata={
            "help": ("The total number of n-best predictions to generate in the nbest_predictions.json output file.")
        },
    )
    null_score_diff_threshold: Optional[float] = field(
        default=0.0,
        metadata={"help": ("If null_score - best_non_null is greater than the threshold predict null.")},
    )
    max_query_length: Optional[int] = field(
        default=64,
        metadata={"help": ("Max query length.")},
    )
    max_answer_length: Optional[int] = field(
        default=65,
        metadata={"help": ("Max answer length.")},
    )
    use_amp: Optional[bool] = field(
        default=False,
        metadata={"help": ("Enable mixed precision training.")},
    )


@dataclass
class DataArguments:
    data_dir: Optional[str] = field(
        default="./data/cmrc2018",
        metadata={"help": ("the path of cmrc2018 data.")},
    )
    save_nbest_json: Optional[bool] = field(
        default=False,
        metadata={"help": ("Enable save nbest json.")},
    )


parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


@dataclass
class Train_DataCollator:
    tokenizer: ChineseBertTokenizer

    def __call__(self, features):
        input_ids = []
        token_type_ids = []
        pinyin_ids = []
        start_positions = []
        end_positions = []
        batch = {}

        for feature in features:
            input_ids.append(feature["input_ids"])
            token_type_ids.append(feature["token_type_ids"])
            pinyin_ids.append(feature["pinyin_ids"])
            start_positions.append(feature["start_positions"])
            end_positions.append(feature["end_positions"])

        input_ids = Pad(axis=0, pad_val=self.tokenizer.pad_token_id)(input_ids)  # input_ids
        token_type_ids = Pad(axis=0, pad_val=0)(token_type_ids)
        pinyin_ids = Pad(axis=0, pad_val=0)(pinyin_ids)  # pinyin_ids
        start_positions = Stack(dtype="int64")(start_positions)
        end_positions = Stack(dtype="int64")(end_positions)

        batch["input_ids"] = input_ids
        batch["token_type_ids"] = token_type_ids
        batch["pinyin_ids"] = pinyin_ids
        batch["start_positions"] = start_positions
        batch["end_positions"] = end_positions

        return batch


@dataclass
class Eval_DataCollator:
    tokenizer: ChineseBertTokenizer

    def __call__(self, features):
        input_ids = []
        token_type_ids = []
        pinyin_ids = []
        batch = {}

        for feature in features:
            input_ids.append(feature["input_ids"])
            token_type_ids.append(feature["token_type_ids"])
            pinyin_ids.append(feature["pinyin_ids"])

        input_ids = Pad(axis=0, pad_val=self.tokenizer.pad_token_id)(input_ids)  # input_ids
        token_type_ids = Pad(axis=0, pad_val=0)(token_type_ids)
        pinyin_ids = Pad(axis=0, pad_val=0)(pinyin_ids)  # pinyin_ids

        batch["input_ids"] = input_ids
        batch["token_type_ids"] = token_type_ids
        batch["pinyin_ids"] = pinyin_ids

        return batch


def compute_metrics(eval_preds, dataloader, args):
    all_start_logits, all_end_logits = eval_preds.predictions
    all_start_logits = all_start_logits.tolist()
    all_end_logits = all_end_logits.tolist()
    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        dataloader.dataset.data,
        dataloader.dataset.new_data,
        (all_start_logits, all_end_logits),
        False,
        model_args.n_best_size,
        model_args.max_answer_length,
        model_args.null_score_diff_threshold,
    )

    save_json(all_predictions, os.path.join(args.output_dir, "all_predictions.json"))
    if data_args.save_nbest_json:
        save_json(all_nbest_json, os.path.join(args.output_dir, "all_nbest_json.json"))

    ground_truth_file = os.path.join(data_args.data_dir, "dev.json")

    eval_results = get_result(
        ground_truth_file=ground_truth_file, prediction_file=os.path.join(args.output_dir, "all_predictions.json")
    )
    print("CMRC2018 EVALUATE.")
    print(eval_results)
    print("SQUAD EVALUATE.")
    squad_evaluate(
        examples=dataloader.dataset.data,
        preds=all_predictions,
        na_probs=scores_diff_json,
    )
    return eval_results


def train():
    model_args.model_type = model_args.model_type.lower()
    training_args.logdir = os.path.join(training_args.output_dir, "logs")
    os.makedirs("caches", exist_ok=True)
    os.makedirs(training_args.logdir, exist_ok=True)

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(training_args.seed)

    # get model and tokenizer
    model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    model = model_class.from_pretrained(model_args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)

    # get dataloader
    args = {}
    args["training_args"] = training_args
    args["data_args"] = data_args
    args["model_args"] = model_args
    train_ds = get_train_dataset(tokenizer, args)
    dev_ds = get_dev_dataset(tokenizer, args)
    train_collator = Train_DataCollator(tokenizer)
    dev_collator = Eval_DataCollator(tokenizer)

    criterion = CrossEntropyLossForSQuAD()

    trainer = EvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=dev_ds if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=train_collator,
        criterion=criterion,
        compute_metrics=compute_metrics,
    )
    trainer.set_eval_collator(dev_collator)

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
    train()
