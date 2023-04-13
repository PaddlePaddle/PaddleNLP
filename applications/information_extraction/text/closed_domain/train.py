# coding=utf-8
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

import argparse
import os
import time

import paddle
from evaluate import evaluate
from utils import create_dataloader, get_eval_golds, get_label_maps, reader, set_seed

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import (
    AutoTokenizer,
    ErnieForClosedDomainIE,
    LinearDecayWithWarmup,
)
from paddlenlp.utils.log import logger


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(args.seed)

    label_maps = get_label_maps(args.label_maps_path)
    golds = get_eval_golds(args.dev_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model = ErnieForClosedDomainIE.from_pretrained(
        args.model_name_or_path,
        entity_id2label=label_maps["entity_id2label"],
        relation_id2label=label_maps["relation_id2label"],
        extraction_schema=label_maps["schema"],
    )

    train_ds = load_dataset(
        reader,
        data_path=args.train_path,
        tokenizer=tokenizer,
        label_maps=label_maps,
        max_seq_len=args.max_seq_len,
        doc_stride=args.doc_stride,
        lazy=False,
    )

    dev_ds = load_dataset(
        reader,
        data_path=args.dev_path,
        tokenizer=tokenizer,
        label_maps=label_maps,
        max_seq_len=args.max_seq_len,
        doc_stride=args.doc_stride,
        lazy=False,
    )

    train_dataloader = create_dataloader(
        train_ds,
        tokenizer=tokenizer,
        label_maps=label_maps,
        batch_size=args.batch_size,
        mode="train",
    )

    dev_dataloader = create_dataloader(
        dev_ds,
        tokenizer=tokenizer,
        label_maps=label_maps,
        batch_size=args.batch_size,
        mode="dev",
    )

    num_training_steps = len(train_dataloader) * args.num_epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    global_step, best_f1 = 1, 0.0
    tr_loss, logging_loss = 0.0, 0.0
    tic_train = time.time()
    for epoch in range(1, args.num_epochs + 1):
        for batch in train_dataloader:
            loss, _ = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss.backward()

            tr_loss += loss.item()

            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step % args.logging_steps == 0 and rank == 0:
                time_diff = time.time() - tic_train
                loss_avg = (tr_loss - logging_loss) / args.logging_steps
                logger.info(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss_avg, args.logging_steps / time_diff)
                )
                logging_loss = tr_loss
                tic_train = time.time()

            if global_step % args.eval_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                model.save_pretrained(save_dir)
                logger.disable()
                tokenizer.save_pretrained(save_dir)
                logger.enable()

                precision, recall, f1 = evaluate(model, dev_dataloader, label_maps, golds)
                logger.info("Evaluation Precision： %.5f, Recall: %.5f, F1: %.5f" % (precision, recall, f1))

                if f1 > best_f1:
                    logger.info(f"best F1 performance has been updated: {best_f1:.5f} --> {f1:.5f}")
                    best_f1 = f1
                    save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    model.save_pretrained(save_dir)
                    logger.disable()
                    tokenizer.save_pretrained(save_dir)
                    logger.enable()
                tic_train = time.time()

            global_step += 1


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="./data/train_data.json", type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default="./data/dev_data.json", type=str, help="The path of dev set.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum input sequence length.")
    parser.add_argument("--doc_stride", type=int, default=256, help="Window size of sliding window.")
    parser.add_argument("--label_maps_path", default="./data/label_maps.json", type=str, help="The file path of the labels dictionary.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay rate for L2 regularizer.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proportion over the training process.")
    parser.add_argument("--num_epochs", default=50, type=int, help="Number of epoches for training.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--model_name_or_path", default="ernie-3.0-base-zh", type=str, help="Select the pretrained model for Global Pointer.")
    parser.add_argument("--logging_steps", default=10, type=int, help="The interval steps to logging.")
    parser.add_argument("--eval_steps", default=100, type=int, help="The interval steps to evaluate model performance.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--init_from_ckpt", default=None, type=str, help="The path of model parameters for initialization.")

    args = parser.parse_args()
    # yapf: enable

    do_train()
