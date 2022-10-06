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
import argparse
import random
import time
import distutils.util
from pprint import pprint
from functools import partial
import numpy as np
from itertools import chain
from datasets import load_dataset
import math
import paddle
import paddle.nn as nn
from paddle.io import BatchSampler, DistributedBatchSampler, DataLoader
from paddlenlp.transformers import CodeGenForCausalLM, CodeGenTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.data import DataCollatorWithPadding
from paddle.metric import Accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_name_or_path",
                        default="Salesforce/codegen-350M-mono",
                        type=str,
                        required=True,
                        help="Path to pre-trained model. ")
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input training data file.")
    parser.add_argument("--validation_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input validation data file.")
    parser.add_argument(
        "--block_size",
        default=None,
        type=int,
        help=
        "The training dataset will be truncated in block of this size for training. "
    )
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--train_batch_size",
        default=20,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=12,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help=
        "Linear warmup over warmup_steps. If > 0: Override warmup_proportion")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Linear warmup proportion over total steps.")
    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument("--overwrite_cache",
                        action="store_true",
                        help="Whether to overwrite cache for dataset.")
    parser.add_argument("--use_amp",
                        default=False,
                        type=distutils.util.strtobool,
                        help="Enable mixed precision training.")
    parser.add_argument("--scale_loss",
                        default=2**15,
                        type=float,
                        help="The value of scale_loss for fp16.")
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
def evaluate(model, data_loader, loss_fct):
    model.eval()
    metric = Accuracy()
    metric.reset()
    losses = []
    model = model._layers if isinstance(model, paddle.DataParallel) else model
    for batch in data_loader:
        labels = batch.pop("labels")
        logits, _ = model(**batch)
        loss = loss_fct(logits[:, :-1, :], labels[:, 1:])
        correct = metric.compute(paddle.max(logits[:, :-1, :], axis=-1),
                                 labels[:, 1:])
        losses.append(loss)
    losses = paddle.concat(losses)
    eval_loss = paddle.mean(losses)
    perplexity = math.exp(eval_loss)
    accuracy = metric.accumulate()
    logger.info("[validation] accuracy: %f, loss: %f, ppl: %f" %
                (accuracy, eval_loss, perplexity))
    model.train()
    return perplexity


def convert_example(examples, tokenizer):
    """convert examples into necessary features"""
    # Convert raw text to feature
    tokenized_examples = tokenizer(examples["code"],
                                   return_attention_mask=True,
                                   return_position_ids=False,
                                   return_token_type_ids=False)
    return tokenized_examples


def group_texts(examples, block_size):
    concatenated_examples = {
        k: list(chain(*examples[k]))
        for k in examples.keys()
    }
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def process_ds(dataset, tokenizer, overwrite_cache, block_size):
    trans_func = partial(convert_example, tokenizer=tokenizer)
    dataset = dataset.map(trans_func,
                          batched=True,
                          remove_columns=dataset.column_names,
                          load_from_cache_file=overwrite_cache)
    trans_func = partial(group_texts, block_size=block_size)
    dataset = dataset.map(trans_func,
                          batched=True,
                          load_from_cache_file=overwrite_cache)
    return dataset


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    tokenizer = CodeGenTokenizer.from_pretrained(args.model_name_or_path)

    train_set = load_dataset("json", data_files=args.train_file, split="train")
    dev_set = load_dataset("json",
                           data_files=args.validation_file,
                           split="train")

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    train_set = process_ds(train_set, tokenizer, args.overwrite_cache,
                           block_size)
    dev_set = process_ds(dev_set, tokenizer, args.overwrite_cache, block_size)

    batchify_fn = DataCollatorWithPadding(tokenizer, return_attention_mask=True)

    train_batch_sampler = DistributedBatchSampler(
        train_set, batch_size=args.train_batch_size, shuffle=True)

    train_data_loader = DataLoader(dataset=train_set,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=batchify_fn,
                                   num_workers=0,
                                   return_list=True)

    dev_batch_sampler = BatchSampler(dev_set,
                                     batch_size=args.eval_batch_size,
                                     shuffle=False)
    dev_data_loader = DataLoader(dataset=dev_set,
                                 batch_sampler=dev_batch_sampler,
                                 collate_fn=batchify_fn,
                                 num_workers=0,
                                 return_list=True)

    model = CodeGenForCausalLM.from_pretrained(args.model_name_or_path)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = math.ceil(num_training_steps /
                                     len(train_data_loader))
    else:
        num_training_steps = len(train_data_loader) * args.num_train_epochs
        num_train_epochs = args.num_train_epochs

    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    loss_fct = nn.CrossEntropyLoss()
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    global_step = 0
    best_eval_ppl = float("inf")
    tic_train = time.time()
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            labels = batch.pop("labels")
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits, _ = model(**batch)
                loss = loss_fct(logits[:, :-1, :], labels[:, 1:])
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
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, ppl: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, math.exp(loss),
                       optimizer.get_lr(), args.logging_steps /
                       (time.time() - tic_train)))
                tic_train = time.time()
            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                ppl = evaluate(model, dev_data_loader, loss_fct)
                logger.info("eval done total : %s s" % (time.time() - tic_eval))
                if best_eval_ppl > ppl and paddle.distributed.get_rank() == 0:
                    best_eval_ppl = ppl
                    output_dir = args.output_dir
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

            if global_step >= num_training_steps:
                break
        if global_step >= num_training_steps:
            break

    if paddle.distributed.get_rank() == 0:
        output_dir = os.path.join(args.output_dir,
                                  "codegen_model_final_%d" % global_step)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Need better way to get inner model of DataParallel
        model_to_save = model._layers if isinstance(
            model, paddle.DataParallel) else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    do_train(args)
