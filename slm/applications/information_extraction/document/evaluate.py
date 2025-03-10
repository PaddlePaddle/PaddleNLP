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

import paddle
from utils import convert_example, reader

from paddlenlp.datasets import MapDataset, load_dataset
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.transformers import UIEX, AutoTokenizer
from paddlenlp.utils.ie_utils import (
    compute_metrics,
    get_relation_type_dict,
    uie_loss_func,
    unify_prompt_name,
)
from paddlenlp.utils.log import logger


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for evaluation.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    test_path: str = field(default=None, metadata={"help": "The path of test set."})

    schema_lang: str = field(
        default="ch", metadata={"help": "Select the language type for schema, such as 'ch', 'en'"}
    )

    max_seq_len: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    debug: bool = field(
        default=False,
        metadata={"help": "Whether choose debug mode."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: Optional[str] = field(
        default=None, metadata={"help": "The path of saved model that you want to load."}
    )


def do_eval():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    model = UIEX.from_pretrained(model_args.model_path)

    test_ds = load_dataset(reader, data_path=data_args.test_path, max_seq_len=data_args.max_seq_len, lazy=False)
    trans_fn = partial(convert_example, tokenizer=tokenizer, max_seq_len=data_args.max_seq_len)
    if data_args.debug:
        class_dict = {}
        relation_data = []

        for data in test_ds:
            class_name = unify_prompt_name(data["prompt"])
            # Only positive examples are evaluated in debug mode
            if len(data["result_list"]) != 0:
                p = "的" if data_args.schema_lang == "ch" else " of "
                if p not in data["prompt"]:
                    class_dict.setdefault(class_name, []).append(data)
                else:
                    relation_data.append((data["prompt"], data))

        relation_type_dict = get_relation_type_dict(relation_data, schema_lang=data_args.schema_lang)
    test_ds = test_ds.map(trans_fn)

    trainer = Trainer(
        model=model,
        criterion=uie_loss_func,
        args=training_args,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    eval_metrics = trainer.evaluate()
    logger.info("-----Evaluate model-------")
    logger.info("Class Name: ALL CLASSES")
    logger.info(
        "Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f"
        % (eval_metrics["eval_precision"], eval_metrics["eval_recall"], eval_metrics["eval_f1"])
    )
    logger.info("-----------------------------")
    if data_args.debug:
        for key in class_dict.keys():
            test_ds = MapDataset(class_dict[key])
            test_ds = test_ds.map(trans_fn)
            eval_metrics = trainer.evaluate(eval_dataset=test_ds)

            logger.info("Class Name: %s" % key)
            logger.info(
                "Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f"
                % (eval_metrics["eval_precision"], eval_metrics["eval_recall"], eval_metrics["eval_f1"])
            )
            logger.info("-----------------------------")
        for key in relation_type_dict.keys():
            test_ds = MapDataset(relation_type_dict[key])
            test_ds = test_ds.map(trans_fn)
            eval_metrics = trainer.evaluate(eval_dataset=test_ds)
            logger.info("-----------------------------")
            if data_args.schema_lang == "ch":
                logger.info("Class Name: X的%s" % key)
            else:
                logger.info("Class Name: %s of X" % key)
            logger.info(
                "Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f"
                % (eval_metrics["eval_precision"], eval_metrics["eval_recall"], eval_metrics["eval_f1"])
            )


if __name__ == "__main__":
    do_eval()
