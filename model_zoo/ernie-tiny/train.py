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
from typing import Optional

import paddle
from model import JointErnie, NLULoss
from utils import compute_metrics, get_label_name, read_example

from paddlenlp.data import DataCollatorForTokenClassification
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    CompressionArguments,
    PdArgumentParser,
    Trainer,
    get_last_checkpoint,
)
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    train_path: str = field(default=None, metadata={"help": "The screen data path for train dataset."})
    dev_path: str = field(
        default=None,
        metadata={"help": "The screen data path for dev dataset. Defaults to None."},
    )
    test_path: str = field(default=None, metadata={"help": "Test data path. Defaults to None."})
    intent_label_path: str = field(default=None, metadata={"help": "Intent label dict path. Defaults to None."})
    slot_label_path: str = field(default=None, metadata={"help": "Slot label dict path. Defaults to None."})
    max_seq_length: Optional[int] = field(
        default=16,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Defaults to 16."
        },
    )
    ignore_index: Optional[int] = field(
        default=9999,
        metadata={
            "help": "Padding index, and it's used to pad noscreen label in screen data, "
            "and pad screen label in noscreen data. Defaults to 9999."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default="ernie-3.0-tiny-nano-v2",
        metadata={"help": "Path to pretrained model. Defaults to 'ernie-3.0-tiny-nano-v2'"},
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate for JointErnie. Defaults to 0.1."})


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, compression_args = parser.parse_args_into_dataclasses()

    paddle.set_device(compression_args.device)

    _, _, intent2id, slot2id = get_label_name(data_args.intent_label_path, data_args.slot_label_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    slot_list = [slot.replace("B-", "") for slot in slot2id]
    slot_list = [slot.replace("I-", "") for slot in slot_list]

    tokenizer.add_tokens(slot_list)
    tokenizer.save_pretrained(compression_args.output_dir)

    train_dataset = load_dataset(
        read_example,
        filename=data_args.train_path,
        intent2id=intent2id,
        slot2id=slot2id,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        lazy=False,
    )

    eval_dataset = load_dataset(
        read_example,
        filename=data_args.train_path,
        intent2id=intent2id,
        slot2id=slot2id,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        lazy=False,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer, label_pad_token_id=0, padding="max_length", max_length=data_args.max_seq_length
    )

    model = JointErnie.from_pretrained(
        model_args.model_name_or_path,
        intent_dim=len(intent2id),
        slot_dim=len(slot2id),
        dropout=model_args.dropout,
    )

    criterion = NLULoss()

    trainer = Trainer(
        model=model,
        args=compression_args,
        data_collator=data_collator,
        train_dataset=train_dataset if compression_args.do_train or compression_args.do_compress else None,
        eval_dataset=eval_dataset if compression_args.do_eval or compression_args.do_compress else None,
        criterion=criterion,
        compute_metrics=compute_metrics,
    )

    compression_args.print_config()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(compression_args.output_dir)
        and compression_args.do_train
        and not compression_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(compression_args.output_dir)
        if last_checkpoint is None and len(os.listdir(compression_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({compression_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and compression_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if compression_args.resume_from_checkpoint is not None:
        checkpoint = compression_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if compression_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
    if compression_args:

        @paddle.no_grad()
        def custom_evaluate(self, model, data_loader):
            model.eval()
            slot_right = 0

            intent_right_no_slot = 0
            sample_num = 0
            for batch in data_loader:
                intent_logits, slot_logits, padding_mask = model(
                    input_ids=batch["input_ids"],
                )
                slot_pred = slot_logits.argmax(axis=-1)
                intent_pred = intent_logits.argmax(axis=-1)

                intent_label = batch["intent_label"]
                slot_label = batch["slot_label"]
                if (intent_label == intent_pred) and (intent_label in (0, 2, 3, 4, 6, 7, 8, 10)):
                    intent_right_no_slot += 1
                elif ((slot_pred == slot_label) | padding_mask).all() == 1:
                    slot_right += 1
                sample_num += 1

            accuracy = (slot_right + intent_right_no_slot) / sample_num * 100

            return accuracy

        trainer.compress(custom_evaluate=custom_evaluate)

    if compression_args.do_export:
        model.eval()
        # convert to static graph with specific input description
        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(shape=[None, None], dtype="int32"),  # input_ids
                paddle.static.InputSpec(shape=[None, None], dtype="int32"),  # segment_ids
            ],
        )
        # save converted static graph model
        paddle.jit.save(model, os.path.join(compression_args.output_dir, "infer_model"))


if __name__ == "__main__":
    main()
