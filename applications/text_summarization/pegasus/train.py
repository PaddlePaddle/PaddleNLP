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
import distutils.util
import math
import os
import random
import time
from functools import partial
from pprint import pprint

import numpy as np
import paddle
from datasets import load_dataset
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from tqdm import tqdm
from utils import compute_metrics, convert_example, main_process_first

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.transformers import (
    LinearDecayWithWarmup,
    PegasusChineseTokenizer,
    PegasusForConditionalGeneration,
)
from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default="IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese",
        type=str,
        help="Path to pre-trained model. ",
    )
    parser.add_argument("--train_file", type=str, required=False, default="data/train.json", help="Train data path.")
    parser.add_argument("--eval_file", type=str, required=False, default="data/test.json", help="Eval data path.")
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_source_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--min_target_length",
        default=0,
        type=int,
        help="The minimum total sequence length for target text when generating. ",
    )
    parser.add_argument(
        "--max_target_length",
        default=64,
        type=int,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--epoch",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--train_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion",
    )
    parser.add_argument(
        "--warmup_proportion", default=0.1, type=float, help="Linear warmup proportion over total steps."
    )
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override epoch.",
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.",
    )
    parser.add_argument(
        "--use_amp", default=False, type=distutils.util.strtobool, help="Enable mixed precision training."
    )
    parser.add_argument("--scale_loss", default=2**15, type=float, help="The value of scale_loss for fp16.")
    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, data_loader, tokenizer, min_target_length, max_target_length):
    model.eval()
    all_preds = []
    all_labels = []
    model = model._layers if isinstance(model, paddle.DataParallel) else model
    for batch in tqdm(data_loader, total=len(data_loader), desc="Eval step"):
        labels = batch.pop("labels").numpy()
        preds = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            min_length=min_target_length,
            max_length=max_target_length,
            use_cache=True,
        )[0]
        all_preds.extend(
            tokenizer.batch_decode(preds.numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        all_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    rougel = compute_metrics(all_preds, all_labels)
    model.train()
    return rougel


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    tokenizer = PegasusChineseTokenizer.from_pretrained(args.model_name_or_path)
    train_set = load_dataset("json", data_files=args.train_file, split="train")
    dev_set = load_dataset("json", data_files=args.eval_file, split="train")
    remove_columns = ["content", "title"]
    trans_func = partial(
        convert_example,
        text_column="content",
        summary_column="title",
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    with main_process_first(desc="train dataset map pre-processing"):
        train_set = train_set.map(trans_func, batched=True, load_from_cache_file=True, remove_columns=remove_columns)
    with main_process_first(desc="dev dataset map pre-processing"):
        dev_set = dev_set.map(trans_func, batched=True, load_from_cache_file=True, remove_columns=remove_columns)

    model = PegasusForConditionalGeneration.from_pretrained(args.model_name_or_path)
    batchify_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    train_batch_sampler = DistributedBatchSampler(train_set, batch_size=args.train_batch_size, shuffle=True)

    train_data_loader = DataLoader(
        dataset=train_set, batch_sampler=train_batch_sampler, num_workers=0, collate_fn=batchify_fn, return_list=True
    )

    dev_batch_sampler = BatchSampler(dev_set, batch_size=args.eval_batch_size, shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_set, batch_sampler=dev_batch_sampler, num_workers=0, collate_fn=batchify_fn, return_list=True
    )
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = math.ceil(num_training_steps / len(train_data_loader))
    else:
        num_training_steps = len(train_data_loader) * args.epoch
        num_train_epochs = args.epoch

    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    global_step = 0
    best_rougel = 0
    tic_train = time.time()
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            with paddle.amp.auto_cast(args.use_amp, custom_white_list=["layer_norm", "softmax", "gelu"]):
                lm_logits, new_cache, loss = model(**batch)
            if args.use_amp:
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.minimize(optimizer, scaled_loss)
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % args.logging_steps == 0:
                logger.info(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (
                        global_step,
                        num_training_steps,
                        epoch,
                        step,
                        paddle.distributed.get_rank(),
                        loss,
                        optimizer.get_lr(),
                        args.logging_steps / (time.time() - tic_train),
                    )
                )
                tic_train = time.time()
            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                rougel = evaluate(model, dev_data_loader, tokenizer, args.min_target_length, args.max_target_length)
                logger.info("eval done total : %s s" % (time.time() - tic_eval))
                if paddle.distributed.get_rank() == 0 and best_rougel < rougel:
                    best_rougel = rougel
                    output_dir = args.output_dir
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
            if global_step >= num_training_steps:
                return


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    do_train(args)
