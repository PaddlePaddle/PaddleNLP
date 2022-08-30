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

import os
import argparse
import random
import math
import time
import distutils.util
from functools import partial
import numpy as np

import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DistributedBatchSampler, DataLoader
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from datasets import load_dataset
from paddle.metric import Accuracy
from paddlenlp.ops.optimizer import layerwise_lr_decay
from paddle.optimizer import AdamW
from paddlenlp.data import DataCollatorWithPadding

all_languages = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
    "ur", "vi", "zh"
]


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_type",
        default=None,
        type=str,
        required=True,
        help="The type of the task to finetune.",
        choices=["cross-lingual-transfer", "translate-train-all"])
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pre-trained model.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="Dropout rate.")
    parser.add_argument(
        "--num_train_epochs",
        default=5,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU/XPU for training.",
    )
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--layerwise_decay",
                        default=0.8,
                        type=float,
                        help="Layerwise decay ratio.")
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
                        default=1e-8,
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
                        type=distutils.util.strtobool,
                        default=False,
                        help="Enable mixed precision training.")
    parser.add_argument("--scale_loss",
                        type=float,
                        default=2**15,
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
def evaluate(model, loss_fct, metric, data_loader, language):
    model.eval()
    metric.reset()
    for batch in data_loader:
        labels = batch.pop("labels")
        logits = model(**batch)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    print("[%s] eval loss: %f, acc: %s, " %
          (language.upper(), loss.numpy(), res),
          end="")
    model.train()


def convert_example(example, tokenizer, max_seq_length=256):
    """convert a example into necessary features"""
    # Convert raw text to feature
    tokenized_example = tokenizer(example["premise"],
                                  text_pair=example["hypothesis"],
                                  max_length=max_seq_length,
                                  padding=False,
                                  truncation=True,
                                  return_position_ids=True,
                                  return_attention_mask=True,
                                  return_token_type_ids=False)
    return tokenized_example


def get_test_dataloader(args, language, batchify_fn, trans_func,
                        remove_columns):
    test_ds = load_dataset("xnli", language, split="test")
    test_ds = test_ds.map(trans_func,
                          batched=True,
                          remove_columns=remove_columns,
                          load_from_cache_file=not args.overwrite_cache)
    test_batch_sampler = BatchSampler(test_ds,
                                      batch_size=args.batch_size,
                                      shuffle=False)
    test_data_loader = DataLoader(dataset=test_ds,
                                  batch_sampler=test_batch_sampler,
                                  collate_fn=batchify_fn,
                                  num_workers=0,
                                  return_list=True)
    return test_data_loader


class XnliDataset(Dataset):
    """
    Make all languages datasets be loaded in lazy mode.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        # Ar language has 2000 empty data.
        self.num_samples = [len(i) for i in datasets]
        self.cumsum_len = np.cumsum(self.num_samples)

    def __getitem__(self, idx):
        language_idx = np.argmax(self.cumsum_len > idx)
        last = language_idx - 1 if language_idx > 0 else language_idx
        sample_idx = idx - self.cumsum_len[last] if idx >= self.cumsum_len[
            last] else idx
        return self.datasets[int(language_idx)][int(sample_idx)]

    def __len__(self):
        return self.cumsum_len[-1]


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)
    remove_columns = ["premise", "hypothesis"]
    if args.task_type == "cross-lingual-transfer":
        train_ds = load_dataset("xnli", "en", split="train")
        train_ds = train_ds.map(trans_func,
                                batched=True,
                                remove_columns=remove_columns,
                                load_from_cache_file=not args.overwrite_cache)
    elif args.task_type == "translate-train-all":
        all_train_ds = []
        for language in all_languages:
            train_ds = load_dataset("xnli", language, split="train")
            all_train_ds.append(
                train_ds.map(trans_func,
                             batched=True,
                             remove_columns=remove_columns,
                             load_from_cache_file=not args.overwrite_cache))
        train_ds = XnliDataset(all_train_ds)
    train_batch_sampler = DistributedBatchSampler(train_ds,
                                                  batch_size=args.batch_size,
                                                  shuffle=True)
    batchify_fn = DataCollatorWithPadding(tokenizer)

    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=batchify_fn,
                                   num_workers=0,
                                   return_list=True)

    num_classes = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=num_classes, dropout=args.dropout)
    n_layers = model.ernie_m.config['num_hidden_layers']
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
    # Construct dict
    name_dict = dict()
    for n, p in model.named_parameters():
        name_dict[p.name] = n

    simple_lr_setting = partial(layerwise_lr_decay, args.layerwise_decay,
                                name_dict, n_layers)

    optimizer = AdamW(learning_rate=lr_scheduler,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=args.adam_epsilon,
                      parameters=model.parameters(),
                      weight_decay=args.weight_decay,
                      apply_decay_param_fun=lambda x: x in decay_params,
                      lr_ratio=simple_lr_setting)

    loss_fct = nn.CrossEntropyLoss()
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    metric = Accuracy()

    global_step = 0
    tic_train = time.time()
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            labels = batch.pop("labels")
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits = model(**batch)
                loss = loss_fct(logits, labels)
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
                print(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                for language in all_languages:
                    tic_eval = time.time()
                    test_data_loader = get_test_dataloader(
                        args, language, batchify_fn, trans_func, remove_columns)
                    evaluate(model, loss_fct, metric, test_data_loader,
                             language)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                if paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(
                        args.output_dir,
                        "ernie_m_ft_model_%d.pdparams" % (global_step))
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
        output_dir = os.path.join(
            args.output_dir, "ernie_m_final_model_%d.pdparams" % global_step)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Need better way to get inner model of DataParallel
        model_to_save = model._layers if isinstance(
            model, paddle.DataParallel) else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


def print_arguments(args):
    """print arguments"""
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_train(args)
