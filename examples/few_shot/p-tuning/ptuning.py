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
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from model import ErnieForPretraining, ErnieMLMCriterion
from data import create_dataloader, transform_fn_dict
from data import convert_example, convert_chid_example
from evaluate import do_evaluate, do_evaluate_chid

# yapf: disable
parser = argparse.ArgumentParser()

parser.add_argument("--task_name", required=True, type=str, help="The task_name to be evaluated")
parser.add_argument("--p_embedding_num", type=int, default=1, help="number of p-embedding")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--save_steps', type=int, default=10000, help="Inteval steps to save checkpoint")
parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")

args = parser.parse_args()
# yapf: enable


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

    label_normalize_json = os.path.join("./label_normalized",
                                        args.task_name + ".json")

    label_norm_dict = None
    with open(label_normalize_json, 'r', encoding="utf-8") as f:
        label_norm_dict = json.load(f)

    convert_example_fn = convert_example if args.task_name != "chid" else convert_chid_example
    evaluate_fn = do_evaluate if args.task_name != "chid" else do_evaluate_chid

    train_ds, dev_ds, public_test_ds = load_dataset("fewclue",
                                                    name=args.task_name,
                                                    splits=("train_0", "dev_0",
                                                            "test_public"))

    # Task related transform operations, eg: numbert label -> text_label, english -> chinese
    transform_fn = partial(transform_fn_dict[args.task_name],
                           label_normalize_dict=label_norm_dict)

    # Some fewshot_learning strategy is defined by transform_fn
    # Note: Set lazy=True to transform example inplace immediately,
    # because transform_fn should only be executed only once when
    # iterate multi-times for train_ds
    train_ds = train_ds.map(transform_fn, lazy=False)
    dev_ds = dev_ds.map(transform_fn, lazy=False)
    public_test_ds = public_test_ds.map(transform_fn, lazy=False)

    model = ErnieForPretraining.from_pretrained('ernie-3.0-medium-zh')

    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')

    if args.task_name != "chid":
        # [src_ids, token_type_ids, masked_positions, masked_lm_labels]
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
            Stack(dtype="int64"),  # masked_lm_labels
        ): [data for data in fn(samples)]
    else:
        # [src_ids, token_type_ids, masked_positions, masked_lm_labels, candidate_labels_ids]
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64"),  # masked_positions
            Stack(dtype="int64"),  # masked_lm_labels
            Stack(dtype="int64"
                  ),  # candidate_labels_ids [candidate_num, label_length]
        ): [data for data in fn(samples)]

    trans_func = partial(convert_example_fn,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         p_embedding_num=args.p_embedding_num)

    train_data_loader = create_dataloader(train_ds,
                                          mode='train',
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

    dev_data_loader = create_dataloader(dev_ds,
                                        mode='eval',
                                        batch_size=args.batch_size,
                                        batchify_fn=batchify_fn,
                                        trans_fn=trans_func)

    public_test_data_loader = create_dataloader(public_test_ds,
                                                mode='eval',
                                                batch_size=args.batch_size,
                                                batchify_fn=batchify_fn,
                                                trans_fn=trans_func)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("warmup from:{}".format(args.init_from_ckpt))

    mlm_loss_fn = ErnieMLMCriterion()
    rdrop_loss = paddlenlp.losses.RDropLoss()

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

    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):

            src_ids = batch[0]
            token_type_ids = batch[1]
            masked_positions = batch[2]
            masked_lm_labels = batch[3]

            prediction_scores = model(input_ids=src_ids,
                                      token_type_ids=token_type_ids,
                                      masked_positions=masked_positions)

            if args.rdrop_coef > 0:
                prediction_scores_2 = model(input_ids=src_ids,
                                            token_type_ids=token_type_ids,
                                            masked_positions=masked_positions)
                ce_loss = (
                    mlm_loss_fn(prediction_scores, masked_lm_labels) +
                    mlm_loss_fn(prediction_scores_2, masked_lm_labels)) * 0.5
                kl_loss = rdrop_loss(prediction_scores, prediction_scores_2)
                loss = ce_loss + kl_loss * args.rdrop_coef
            else:
                loss = mlm_loss_fn(prediction_scores, masked_lm_labels)

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, 10 /
                       (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        dev_accuracy, total_num = evaluate_fn(model, tokenizer, dev_data_loader,
                                              label_norm_dict)
        print("epoch:{}, dev_accuracy:{:.3f}, total_num:{}".format(
            epoch, dev_accuracy, total_num))
        test_accuracy, total_num = evaluate_fn(model, tokenizer,
                                               public_test_data_loader,
                                               label_norm_dict)
        print("epoch:{}, test_accuracy:{:.3f}, total_num:{}".format(
            epoch, test_accuracy, total_num))

        if rank == 0:
            save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_param_path = os.path.join(save_dir, 'model_state.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    do_train()
