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
import sys
import os
import random
import time
import logging
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger

from ance.model import SemanticIndexANCE
from data import read_text_pair, read_text_triplet
from data import convert_example, create_dataloader
from data import get_latest_checkpoint, get_latest_ann_data

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoints', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--ann_data_dir", default='./ann_data', type=str, help="The output directory where the ann generated training data will be saved.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--max_training_steps", default=1000000, type=int, help="The maximum total steps for training")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--output_emb_size", default=None, type=int, help="output_embedding_size")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--save_steps', type=int, default=10000, help="Inteval steps to save checkpoint")
parser.add_argument("--train_set_file", type=str, required=True, help="The full path of train_set_file")
parser.add_argument("--margin", default=0.3, type=float, help="Margin for pair-wise margin_rank_loss")


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

    pretrained_model = AutoModel.from_pretrained('ernie-3.0-medium-zh')

    latest_checkpoint, latest_global_step = get_latest_checkpoint(args)
    logger.info("get latest_checkpoint:{}".format(latest_checkpoint))

    model = SemanticIndexANCE(pretrained_model,
                              margin=args.margin,
                              output_emb_size=args.output_emb_size)

    if latest_checkpoint:
        state_dict = paddle.load(latest_checkpoint)
        model.set_dict(state_dict)
        print("warmup from:{}".format(latest_checkpoint))

    model = paddle.DataParallel(model)

    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # pos_sample_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # pos_sample_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # neg_sample_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # neg_sample_segment
    ): [data for data in fn(samples)]

    global_step = 0

    while global_step < args.max_training_steps:
        latest_ann_data, latest_ann_data_step = get_latest_ann_data(
            args.ann_data_dir)

        if latest_ann_data_step == -1:
            # No ann_data generated yet
            latest_ann_data = args.train_set_file
            logger.info("No ann_data generated yet, Use training_set:{}".format(
                args.train_set_file))
        else:
            # Using ann_data to training model
            logger.info("Latest ann_data is ready for training: [{}]".format(
                latest_ann_data))

        train_ds = load_dataset(read_text_triplet,
                                data_path=latest_ann_data,
                                lazy=False)

        train_data_loader = create_dataloader(train_ds,
                                              mode='train',
                                              batch_size=args.batch_size,
                                              batchify_fn=batchify_fn,
                                              trans_fn=trans_func)

        num_training_steps = len(train_data_loader) * args.epochs

        lr_scheduler = LinearDecayWithWarmup(args.learning_rate,
                                             num_training_steps,
                                             args.warmup_proportion)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=clip)

        tic_train = time.time()
        for epoch in range(1, args.epochs + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                text_input_ids, text_token_type_ids, pos_sample_input_ids, pos_sample_token_type_ids, neg_sample_input_ids, neg_sample_token_type_ids, = batch

                loss = model(
                    text_input_ids=text_input_ids,
                    pos_sample_input_ids=pos_sample_input_ids,
                    neg_sample_input_ids=neg_sample_input_ids,
                    text_token_type_ids=text_token_type_ids,
                    pos_sample_token_type_ids=pos_sample_token_type_ids,
                    neg_sample_token_type_ids=neg_sample_token_type_ids)

                global_step += 1
                if global_step % 10 == 0 and rank == 0:
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, trainning_file: %s"
                        % (global_step, epoch, step, loss, 10 /
                           (time.time() - tic_train), latest_ann_data))
                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % args.save_steps == 0 and rank == 0:
                    save_dir = os.path.join(args.save_dir, str(global_step))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir,
                                                   'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)

                    # Flag to indicate succeefully save model
                    succeed_flag_file = os.path.join(save_dir,
                                                     "succeed_flag_file")
                    open(succeed_flag_file, 'a').close()


if __name__ == "__main__":
    do_train()
