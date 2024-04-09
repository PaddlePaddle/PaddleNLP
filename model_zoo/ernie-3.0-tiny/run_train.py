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
    cut_embeddings,
    get_last_checkpoint,
)
from paddlenlp.transformers import AutoTokenizer, ErnieConfig
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
    max_vocab_size: Optional[int] = field(
        default=8000,
        metadata={"help": "The Maximum vocab size after pruning word embeddings. Defaults to 8000."},
    )
    ignore_index: Optional[int] = field(
        default=0,
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
        default="ernie-3.0-tiny-nano-v2-zh",
        metadata={"help": "Path to pretrained model. Defaults to 'ernie-3.0-tiny-nano-v2-zh'"},
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate for JointErnie. Defaults to 0.1."})


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, compression_args = parser.parse_args_into_dataclasses()

    paddle.set_device(compression_args.device)

    _, _, intent2id, slot2id = get_label_name(data_args.intent_label_path, data_args.slot_label_path)

    model = JointErnie.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        intent_dim=len(intent2id),
        slot_dim=len(slot2id),
        dropout=model_args.dropout,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    slot_list = [slot.replace("B-", "") for slot in slot2id]
    slot_list = [slot.replace("I-", "") for slot in slot_list]
    slot_list += ["<", ">", "/"]

    if compression_args.prune_embeddings and compression_args.do_train:
        filelist = [data_args.train_path, data_args.dev_path]
        vocab_dict = {}
        for i in range(tokenizer.vocab_size):
            vocab_dict[i] = 0
        max_freq = 0

        for filename in filelist:
            f = open(filename)
            for line in f:
                if len(line.strip().split("\t")) < 2:
                    continue
                idx_list = tokenizer(line.strip().split("\t")[1])["input_ids"]
                for idx in idx_list:
                    if idx in vocab_dict:
                        vocab_dict[idx] += 1
                    else:
                        vocab_dict[idx] = 0
                    max_freq = max(max_freq, vocab_dict[idx])
            f.close()
        for special_token in tokenizer.all_special_tokens:
            if special_token == "[PAD]":
                vocab_dict[tokenizer.convert_tokens_to_ids([special_token])[0]] = max_freq + 2
            else:
                vocab_dict[tokenizer.convert_tokens_to_ids([special_token])[0]] = max_freq + 1

        vocab_dict = sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True)

        vocab_dict = vocab_dict[: data_args.max_vocab_size - len(slot_list)]
        word_emb_index = [vocab[0] for vocab in vocab_dict]

        config = ErnieConfig.from_pretrained(model_args.model_name_or_path)
        pretrained_model_dir = os.path.join(compression_args.output_dir, "pretrained_model")
        # Rewrites model, tokenizer and pretrained_model directory.
        cut_embeddings(
            model,
            tokenizer,
            config,
            word_emb_index,
            data_args.max_seq_length,
            data_args.max_vocab_size,
            pretrained_model_dir,
        )

        # Reloads model and tokenizer
        model = JointErnie.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_dir,
            intent_dim=len(intent2id),
            slot_dim=len(slot2id),
            dropout=model_args.dropout,
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)

    tokenizer.add_tokens(slot_list)

    train_dataset = load_dataset(
        read_example,
        filename=data_args.train_path,
        intent2id=intent2id,
        slot2id=slot2id,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        no_entity_id=data_args.ignore_index,
        lazy=False,
    )

    eval_dataset = load_dataset(
        read_example,
        filename=data_args.dev_path,
        intent2id=intent2id,
        slot2id=slot2id,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        no_entity_id=data_args.ignore_index,
        lazy=False,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer, label_pad_token_id=0, padding="max_length", max_length=data_args.max_seq_length
    )

    criterion = NLULoss()

    trainer = Trainer(
        model=model,
        args=compression_args,
        data_collator=data_collator,
        train_dataset=train_dataset if compression_args.do_train or compression_args.do_compress else None,
        eval_dataset=eval_dataset if compression_args.do_eval or compression_args.do_compress else None,
        criterion=criterion,
        tokenizer=tokenizer,
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

    if compression_args.do_eval:
        trainer.evaluate()
    if compression_args.do_compress:

        @paddle.no_grad()
        def custom_evaluate(self, model, data_loader):
            model.eval()
            intent_right, slot_right, sample_num = 0, 0, 0
            for batch in data_loader:
                logits = model(input_ids=batch["input_ids"])

                if len(logits) == 2:
                    intent_logits, slot_logits, padding_mask = logits[0]
                elif len(logits) == 3:
                    intent_logits, slot_logits, padding_mask = logits

                slot_pred = slot_logits.argmax(axis=-1)
                intent_pred = intent_logits.argmax(axis=-1)

                intent_label = batch["intent_label"]
                slot_label = batch["slot_label"]

                batch_num = intent_label.shape[0]
                for i in range(batch_num):
                    if intent_label[i] == intent_pred[i]:
                        intent_right += 1
                        if intent_label[i] in (0, 2, 3, 4, 6, 7, 8, 10):
                            slot_right += 1
                        elif paddle.all((slot_pred[i] == slot_label[i]) | padding_mask[i]):
                            slot_right += 1
                sample_num += batch_num

            intent_accuracy = intent_right / sample_num * 100
            accuracy = slot_right / sample_num * 100
            logger.info("accuray: %.2f, intent_accuracy: %.2f" % (accuracy, intent_accuracy))
            model.train()

            return accuracy

        trainer.compress(custom_evaluate=custom_evaluate)

    if compression_args.do_export:
        model.eval()
        # convert to static graph with specific input description
        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(shape=[None, None], dtype=compression_args.input_dtype),  # input_ids
            ],
        )
        # save converted static graph model
        paddle.jit.save(model, os.path.join(compression_args.output_dir, "infer_model"))
        tokenizer.save_pretrained(compression_args.output_dir)


if __name__ == "__main__":
    main()
