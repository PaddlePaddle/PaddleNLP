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

import os
from dataclasses import dataclass, field, fields
from functools import partial

import paddle
import paddle.nn as nn
from data import cnn_dm_convert_example, load_local_dataset
from utils import GLMTrainer

from paddlenlp.metrics import Rouge1, Rouge2, RougeL
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer


# yafp: disable
@dataclass
class DataArgument:
    data_path: str = field(default="./data", metadata={"help": "The path of dataset."})
    task_name: str = field(default="cnn_dm", metadata={"help": "The name of task."})
    src_length: int = field(default=608, metadata={"help": "The max length of source text."})
    tgt_length: int = field(default=160, metadata={"help": "The max length of target text."})
    min_tgt_length: int = field(default=55, metadata={"help": "The min length of target text."})
    out_seq_length: int = field(default=256, metadata={"help": "The length of decoded prediction."})
    length_penalty: float = field(default=0.7, metadata={"help": "The length penalty."})
    no_repeat_ngram_size: int = field(default=3, metadata={"help": "The no repeat ngram size."})
    num_beams: int = field(default=5, metadata={"help": "The number of beams."})
    select_topk: bool = field(default=True, metadata={"help": "Whether to select top k tokens for generation."})
    top_p: float = field(default=0.0)
    top_k: int = field(default=0)
    no_block_position: bool = field(default=False)


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default="glm-2b", metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    label_smoothing: float = field(default=0.1, metadata={"help": "The label smoothing parameter."})


# yafp: enable


def main():
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    for field_item in fields(model_args):
        setattr(training_args, field_item.name, getattr(model_args, field_item.name))
    for field_item in fields(data_args):
        setattr(training_args, field_item.name, getattr(data_args, field_item.name))

    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    # TODO: FP16, DDP
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, output_predict=True, parallel_output=True
    )
    # TODO: prepare_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load the dataset.
    train_ds, dev_ds, test_ds = load_local_dataset(data_path=data_args.data_path, splits=["train", "dev", "test"])
    trans_func = partial(cnn_dm_convert_example, tokenizer=tokenizer, data_args=data_args)
    train_ds = train_ds.map(partial(trans_func, is_test=False))
    dev_ds = dev_ds.map(trans_func)
    test_ds = test_ds.map(trans_func)

    # TODO: Set seed for sampler based on specific epoch and seed.
    criterion = nn.loss.CrossEntropyLoss(reduction="none")

    def compute_metrics(eval_preds):
        rouge1 = Rouge1()
        rouge2 = Rouge2()
        rougel = RougeL()
        rouge1_score = rouge1.score(eval_preds.predictions, eval_preds.label_ids)
        rouge2_score = rouge2.score(eval_preds.predictions, eval_preds.label_ids)
        rougel_score = rougel.score(eval_preds.predictions, eval_preds.label_ids)
        return {
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougel": rougel_score,
        }

    trainer = GLMTrainer(
        model=model,
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        do_generation=True,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        # trainer.save_metrics("eval", eval_result.metrics)
        print(eval_result)

    if training_args.do_export:
        export_path = os.path.join(training_args.output_dir, "export")
        trainer.export_model(export_path=export_path)


if __name__ == "__main__":
    main()
