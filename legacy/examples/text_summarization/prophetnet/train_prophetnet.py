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

from dataclasses import dataclass, field
from typing import Optional

import paddle
from tqdm import tqdm

from paddlenlp.data import Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments, set_seed
from paddlenlp.transformers.prophetnet.modeling import (
    ProphetNetForConditionalGeneration,
)
from paddlenlp.transformers.prophetnet.tokenizer import ProphetNetTokenizer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="prophetnet-large-uncased",
        metadata={"help": ("Path to pre-trained model.")},
    )
    warmup_init_lr: Optional[float] = field(
        default=1e-07,
    )


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default="gigaword",
        metadata={"help": ("Path to tokenizer vocab file.")},
    )


def read(data_path):
    data_path_src = data_path[0]
    data_path_tgt = data_path[1]
    with open(data_path_src, "r", encoding="utf-8") as f_d_s:
        src_lines_length = len(f_d_s.readlines())
    with open(data_path_tgt, "r", encoding="utf-8") as f_d_t:
        tgt_lines_length = len(f_d_t.readlines())
    assert src_lines_length == tgt_lines_length
    with open(data_path_src, "r", encoding="utf-8") as f_d_s:
        with open(data_path_tgt, "r", encoding="utf-8") as f_d_t:
            for row_d_s, row_d_t in tqdm(zip(f_d_s, f_d_t), total=src_lines_length):
                yield {"article": row_d_s, "highlights": row_d_t}


class InverseSquareRootSchedule(paddle.optimizer.lr.LRScheduler):
    def __init__(self, warmup_init_lr, warmup_end_lr, warmup_steps, last_epoch=-1, verbose=False):
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_steps
        self.decay_factor = warmup_end_lr * warmup_steps**0.5
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        super(InverseSquareRootSchedule, self).__init__(warmup_init_lr, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            self.base_lr = self.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            self.base_lr = self.decay_factor * self.last_epoch**-0.5
        return self.base_lr


def convert_example(is_test=False):
    def warpper(example):
        """convert an example into necessary features"""
        tokens = example["article"]
        labels = example["highlights"]
        src_ids, src_attention_mask_ids = tokens.split("$1$")
        src_ids = [int(i) for i in src_ids.split(" ")]
        src_attention_mask_ids = [int(i) for i in src_attention_mask_ids.split(" ")]

        if not is_test:
            labels, decoder_input_attention_mask_ids = labels.split("$1$")
            labels = [int(i) for i in labels.split(" ")]
            decoder_input_attention_mask_ids = [int(i) for i in decoder_input_attention_mask_ids.split(" ")]
            decoder_input_ids = [labels[-1]] + labels[:-1]
            return src_ids, src_attention_mask_ids, decoder_input_ids, decoder_input_attention_mask_ids, labels

        else:
            return src_ids, src_attention_mask_ids

    return warpper


@dataclass
class DataCollator:
    tokenizer: ProphetNetTokenizer

    def __call__(self, features):
        src_ids = []
        src_pids = []
        tgt_ids = []
        tgt_pids = []
        labels = []
        batch = {}

        for feature in features:
            src_idx, src_pid, tgt_idx, tgt_pid, label = feature
            src_ids.append(src_idx)
            src_pids.append(src_pid)
            tgt_ids.append(tgt_idx)
            tgt_pids.append(tgt_pid)
            labels.append(label)

        src_ids = (Pad(axis=0, pad_val=self.tokenizer.pad_token_id)(src_ids),)
        src_pids = (Pad(axis=0, pad_val=0)(src_pids),)
        tgt_ids = (Pad(axis=0, pad_val=self.tokenizer.pad_token_id)(tgt_ids),)
        tgt_pids = (Pad(axis=0, pad_val=0)(tgt_pids),)
        labels = (Pad(axis=0, pad_val=self.tokenizer.pad_token_id)(labels),)

        batch["src_ids"] = src_ids[0]
        batch["src_pids"] = src_pids[0]
        batch["tgt_ids"] = tgt_ids[0]
        batch["tgt_pids"] = tgt_pids[0]
        batch["labels"] = labels[0]

        return batch


def loss_func(model, logits, labels, ignore_index=-100):
    expend_targets = paddle.cast(
        paddle.zeros((model.prophetnet.config["ngram"], labels.shape[0], labels.shape[1])).fill_(ignore_index),
        dtype=paddle.int32,
    )

    for i in range(model.prophetnet.config["ngram"]):
        if i > 0 and model.prophetnet.disable_ngram_loss:
            break
        expend_targets[i, :, :] = labels.cast(dtype=paddle.int32)  # B,Ngram,Seq

    logits = logits.transpose([1, 0, 2, 3])

    if model.prophetnet.eps > 0.0:
        expend_targets_mask = paddle.cast(expend_targets != ignore_index, dtype=paddle.float32)
        expend_targets = paddle.nn.functional.one_hot(expend_targets, num_classes=model.vocab_size)
        expend_targets = paddle.nn.functional.label_smooth(expend_targets, epsilon=model.prophetnet.eps)
        loss = paddle.nn.functional.cross_entropy(logits, expend_targets, soft_label=True, reduction="none").squeeze()
        loss = paddle.sum(expend_targets_mask * loss) / expend_targets_mask.sum()
    else:
        loss = paddle.nn.functional.cross_entropy(
            logits, expend_targets.cast(dtype=paddle.int64), ignore_index=ignore_index
        )

    return loss


class ProphetnetTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        src_ids, src_attention_mask_ids, decoder_input_ids, decoder_input_attention_mask_ids, label_ids = inputs
        src_ids = inputs["src_ids"]
        src_attention_mask_ids = inputs["src_pids"]
        decoder_input_ids = inputs["tgt_ids"]
        decoder_input_attention_mask_ids = inputs["tgt_pids"]
        label_ids = inputs["labels"]

        src_ids = src_ids.cast(dtype=paddle.int32)
        src_attention_mask_ids = src_attention_mask_ids.cast(dtype=paddle.int32)
        decoder_input_ids = decoder_input_ids.cast(dtype=paddle.int32)
        decoder_input_attention_mask_ids = decoder_input_attention_mask_ids.cast(dtype=paddle.int32)
        label_ids = label_ids.cast(dtype=paddle.int64)

        outputs = model(
            input_ids=src_ids,
            attention_mask=src_attention_mask_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_input_attention_mask_ids,
        )
        loss = loss_func(model, outputs[2], label_ids, ignore_index=model.padding_idx)

        return (loss, outputs) if return_outputs else loss


def do_train():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(training_args.seed)

    train_data_src = "data/" + data_args.dataset + "_data/uncased_tok_data/train.src"
    train_data_tgt = "data/" + data_args.dataset + "_data/uncased_tok_data/train.tgt"

    dev_data_src = "data/" + data_args.dataset + "_data/uncased_tok_data/dev.src"
    dev_data_tgt = "data/" + data_args.dataset + "_data/uncased_tok_data/dev.tgt"

    train_dataset = load_dataset(read, data_path=[train_data_src, train_data_tgt], lazy=False)
    dev_dataset = load_dataset(read, data_path=[dev_data_src, dev_data_tgt], lazy=False)

    tokenizer = ProphetNetTokenizer.from_pretrained(model_args.model_name_or_path)

    trans_func = convert_example()

    train_dataset = train_dataset.map(trans_func)
    dev_dataset = dev_dataset.map(trans_func)
    batchify_fn = DataCollator(tokenizer)

    model = ProphetNetForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    lr_scheduler = InverseSquareRootSchedule(
        model_args.warmup_init_lr, training_args.learning_rate, training_args.warmup_steps
    )
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=training_args.weight_decay,
        grad_clip=paddle.nn.ClipGradByNorm(training_args.max_grad_norm),
    )

    trainer = ProphetnetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=dev_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=batchify_fn,
        optimizers=(optimizer, lr_scheduler),
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
