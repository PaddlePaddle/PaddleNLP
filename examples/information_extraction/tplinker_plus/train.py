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
import json
import time
import os
from functools import partial

import paddle
import paddle.nn as nn
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer, AutoModel
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger

from evaluate import evaluate
from criterion import Criterion
from utils import create_dataloader, reader, set_seed, get_re_label_dict
from model import TPLinkerPlus


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(args.seed)

    label_dicts = get_re_label_dict(args.schema_path)
    rel2id = label_dicts["rel2id"]

    train_ds = load_dataset(reader, data_path=args.train_path, lazy=False)
    dev_ds = load_dataset(reader, data_path=args.dev_path, lazy=False)

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")

    train_dataloader = create_dataloader(train_ds,
                                         tokenizer,
                                         max_seq_len=args.max_seq_len,
                                         batch_size=args.batch_size,
                                         is_train=True,
                                         rel2id=rel2id)

    dev_dataloader = create_dataloader(dev_ds,
                                       tokenizer,
                                       max_seq_len=args.max_seq_len,
                                       batch_size=args.batch_size,
                                       is_train=False,
                                       rel2id=rel2id)

    encoder = AutoModel.from_pretrained("ernie-3.0-base-zh")
    model = TPLinkerPlus(encoder, rel2id, shaking_type="cln")

    num_training_steps = len(train_dataloader) * args.num_epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm", "LayerNorm.weight"])
    ]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=grad_clip)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    criterion = Criterion(ghm=False)

    global_step, best_f1 = 1, 0.
    tr_loss, logging_loss = 0.0, 0.0
    tic_train = time.time()
    for epoch in range(1, args.num_epochs + 1):
        for batch in train_dataloader:
            input_ids, attention_masks, labels = batch
            logits = model(input_ids, attention_masks)
            loss = criterion(logits, labels)

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()
            tr_loss += loss.item()

            if global_step % args.logging_steps == 0 and rank == 0:
                time_diff = time.time() - tic_train
                loss_avg = (tr_loss - logging_loss) / args.logging_steps
                logger.info(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss_avg,
                       args.logging_steps / time_diff))
                logging_loss = tr_loss
                tic_train = time.time()

            if global_step % args.valid_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, "model_state.pdparams")
                paddle.save(model.state_dict(), save_param_path)

                precision, recall, f1 = evaluate(model, dev_dataloader,
                                                 label_dicts)
                logger.info(
                    "Evaluation precision: %.5f, recall: %.5f, F1: %.5f" %
                    (precision, recall, f1))

                if f1 > best_f1:
                    logger.info(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    best_f1 = f1
                    save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir,
                                                   "model_state.pdparams")
                    paddle.save(model.state_dict(), save_param_path)
                tic_train = time.time()

            global_step += 1


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_path", default="./data/train_data.json", type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default="./data/dev_data.json", type=str, help="The path of dev set.")
    parser.add_argument("--schema_path", default="./data/all_50_schemas", type=str, help="The file path of the schema for extraction.")
    parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum input sequence length. "
        "Sequences longer than this will be split automatically.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max grad norm to clip gradient.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay rate for L2 regularizer.")
    parser.add_argument("--warmup_proportion", default=0.05, type=float, help="Linear warmup proption over the training process.")
    parser.add_argument("--num_epochs", default=3, type=int, help="Number of epoches for training.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=10, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=100, type=int, help="The interval steps to evaluate model performance.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--init_from_ckpt", default=None, type=str, help="The path of model parameters for initialization.")

    args = parser.parse_args()
    # yapf: enable

    do_train()
