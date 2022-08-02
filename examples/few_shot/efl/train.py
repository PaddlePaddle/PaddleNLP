# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import random
import time
import json
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import create_dataloader, convert_example, processor_dict
from evaluate import do_evaluate
from predict import do_predict, write_fn, predict_file
from task_label_description import TASK_LABELS_DESC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        required=True,
                        type=str,
                        help="The task_name to be evaluated")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--negative_num",
                        default=1,
                        type=int,
                        help="Random negative sample number for efl strategy")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--save_dir",
        default='./checkpoint',
        type=str,
        help="The output directory where the model checkpoints will be written."
    )
    parser.add_argument(
        "--output_dir",
        default='./predict_output',
        type=str,
        help="The output directory where the model checkpoints will be written."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.0,
        type=float,
        help="Linear warmup proption over the training process.")
    parser.add_argument("--init_from_ckpt",
                        type=str,
                        default=None,
                        help="The path of checkpoint to be loaded.")
    parser.add_argument("--seed",
                        type=int,
                        default=1000,
                        help="random seed for initialization")
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument('--save_steps',
                        type=int,
                        default=100000,
                        help="Inteval steps to save checkpoint")
    parser.add_argument(
        "--rdrop_coef",
        default=0.0,
        type=float,
        help=
        "The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works"
    )

    return parser.parse_args()


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_ds, public_test_ds, test_ds = load_dataset("fewclue",
                                                     name=args.task_name,
                                                     splits=("train_0",
                                                             "test_public",
                                                             "test"))

    model = AutoModelForSequenceClassification.from_pretrained(
        'ernie-3.0-medium-zh', num_classes=2)
    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')

    processor = processor_dict[args.task_name](args.negative_num)
    train_ds = processor.get_train_datasets(train_ds,
                                            TASK_LABELS_DESC[args.task_name])

    public_test_ds = processor.get_dev_datasets(
        public_test_ds, TASK_LABELS_DESC[args.task_name])
    test_ds = processor.get_test_datasets(test_ds,
                                          TASK_LABELS_DESC[args.task_name])

    # [src_ids, token_type_ids, labels]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(dtype="int64"),  # labels
    ): [data for data in fn(samples)]

    # [src_ids, token_type_ids]
    predict_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    ): [data for data in fn(samples)]

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)

    predict_trans_func = partial(convert_example,
                                 tokenizer=tokenizer,
                                 max_seq_length=args.max_seq_length,
                                 is_test=True)

    train_data_loader = create_dataloader(train_ds,
                                          mode='train',
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

    public_test_data_loader = create_dataloader(public_test_ds,
                                                mode='eval',
                                                batch_size=args.batch_size,
                                                batchify_fn=batchify_fn,
                                                trans_fn=trans_func)

    test_data_loader = create_dataloader(test_ds,
                                         mode='eval',
                                         batch_size=args.batch_size,
                                         batchify_fn=predict_batchify_fn,
                                         trans_fn=predict_trans_func)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("warmup from:{}".format(args.init_from_ckpt))

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    rdrop_loss = paddlenlp.losses.RDropLoss()
    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):

            src_ids, token_type_ids, labels = batch

            prediction_scores = model(input_ids=src_ids,
                                      token_type_ids=token_type_ids)

            if args.rdrop_coef > 0:
                prediction_scores_2 = model(input_ids=src_ids,
                                            token_type_ids=token_type_ids)
                ce_loss = (criterion(prediction_scores, labels) +
                           criterion(prediction_scores_2, labels)) * 0.5
                kl_loss = rdrop_loss(prediction_scores, prediction_scores_2)
                loss = ce_loss + kl_loss * args.rdrop_coef
            else:
                loss = criterion(prediction_scores, labels)

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, 10 /
                       (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.save_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_dir)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        test_public_accuracy, total_num = do_evaluate(
            model,
            tokenizer,
            public_test_data_loader,
            task_label_description=TASK_LABELS_DESC[args.task_name])

        print("epoch:{}, dev_accuracy:{:.3f}, total_num:{}".format(
            epoch, test_public_accuracy, total_num))

        y_pred_labels = do_predict(
            model,
            tokenizer,
            test_data_loader,
            task_label_description=TASK_LABELS_DESC[args.task_name])

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_file = os.path.join(args.output_dir,
                                   str(epoch) + predict_file[args.task_name])

        write_fn[args.task_name](args.task_name, output_file, y_pred_labels)

        if rank == 0:
            save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_param_path = os.path.join(save_dir, 'model_state.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    args = parse_args()
    do_train()
