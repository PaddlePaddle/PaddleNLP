# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import random
import time
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from data import convert_example, create_dataloader, read_data

from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--train_set", type=str, required=True, help="The full path of train_set_file.")
parser.add_argument("--test_file", type=str, required=True, help="The full path of test file")

parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proportion over the training process.")
parser.add_argument("--valid_steps", default=100, type=int, help="The interval steps to evaluate model performance.")
parser.add_argument("--save_steps", default=100, type=int, help="The interval steps to save checkppoints.")
parser.add_argument("--logging_steps", default=10, type=int, help="The interval steps to logging.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--use_amp", type=strtobool, default=False, help="Enable mixed precision training.")
parser.add_argument("--scale_loss", type=float, default=2**15, help="The value of scale_loss for fp16.")
parser.add_argument('--model_name_or_path', default="rocketqa-base-cross-encoder", help="The pretrained model used for training")
parser.add_argument("--eval_step", default=200, type=int, help="Step interval for evaluation.")
args = parser.parse_args()
# yapf: enable


@paddle.no_grad()
def evaluate(model, metric, data_loader, phase="dev"):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()

    for idx, batch in enumerate(data_loader):
        input_ids, token_type_ids, labels = batch

        pos_probs = model(input_ids=input_ids, token_type_ids=token_type_ids)

        sim_score = F.softmax(pos_probs)

        metric.update(preds=sim_score.numpy(), labels=labels)

    print("eval_{} auc:{:.3}".format(phase, metric.accumulate()))
    metric.reset()
    model.train()


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
    dev_count = paddle.distributed.get_world_size()
    set_seed(args.seed)

    train_ds = load_dataset(read_data, data_path=args.train_set, lazy=False)
    dev_ds = load_dataset(read_data, data_path=args.test_file, lazy=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_classes=2)

    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=args.max_seq_length, is_pair=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # segment
        Stack(dtype="int64"),  # label
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(
        train_ds, mode="train", batch_size=args.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )
    dev_data_loader = create_dataloader(
        dev_ds, mode="dev", batch_size=args.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    model = paddle.DataParallel(model)

    num_training_examples = len(train_ds)
    # 4卡 gpu
    max_train_steps = args.epochs * num_training_examples // args.batch_size // dev_count

    warmup_steps = int(max_train_steps * args.warmup_proportion)

    print("Device count: %d" % dev_count)
    print("Num train examples: %d" % num_training_examples)
    print("Max train steps: %d" % max_train_steps)
    print("Num warmup steps: %d" % warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
    )

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Auc()
    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            acc = paddle.metric.accuracy(input=probs, label=labels)
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % args.logging_steps == 0 and rank == 0:
                time_diff = time.time() - tic_train
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accuracy: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc, args.logging_steps / time_diff)
                )
                tic_train = time.time()
            if global_step % args.eval_step == 0 and rank == 0:
                evaluate(model, metric, dev_data_loader, "dev")
            if global_step % args.save_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model
                model_to_save.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                tic_train = time.time()

    # save final checkpoint
    save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
    model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model
    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    do_train()
