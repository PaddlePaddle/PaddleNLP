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
import distutils.util

import numpy as np
import paddle
from paddle.io import DataLoader
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.experimental import FasterErnieForTokenClassification, to_tensor

parser = argparse.ArgumentParser()

# yapf: disable
parser.add_argument("--save_dir", default="ckpt", type=str, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.", )
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--use_amp", type=distutils.util.strtobool, default=False, help="Enable mixed precision training.")
parser.add_argument("--scale_loss", type=float, default=2**15, help="The value of scale_loss for fp16.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, label_num):
    model.eval()
    metric.reset()
    avg_loss, precision, recall, f1_score = 0, 0, 0, 0
    for texts, labels, seq_lens in data_loader:
        texts = to_tensor(texts)
        logits, preds = model(texts)
        loss = criterion(logits, labels)
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            seq_lens, preds, labels)
        metric.update(num_infer_chunks.numpy(),
                      num_label_chunks.numpy(), num_correct_chunks.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval loss: %f, precision: %f, recall: %f, f1: %f" %
          (avg_loss, precision, recall, f1_score))
    model.train()


def batchify_fn(batch, no_entity_id, ignore_label=-100, max_seq_len=512):
    texts, labels, seq_lens = [], [], []
    # 2 for [CLS] and [SEP]
    batch_max_seq = max([len(example["tokens"]) for example in batch]) + 2
    #  Truncation: Handle max sequence length
    #  If max_seq_len == 0, then do nothing and keep the real length.
    #  If max_seq_len > 0 and
    #  all the input sequence len is over the max_seq_len,
    #  then we truncate it.
    if max_seq_len > 0:
        batch_max_seq = min(batch_max_seq, max_seq_len)
    for example in batch:
        texts.append("".join(example["tokens"]))
        label = example["labels"]
        # 2 for [CLS] and [SEP]
        if len(label) > batch_max_seq - 2:
            label = label[:(batch_max_seq - 2)]
        label = [no_entity_id] + label + [no_entity_id]
        seq_lens.append(len(label))
        if len(label) < batch_max_seq:
            label += [ignore_label] * (batch_max_seq - len(label))
        labels.append(label)
    labels = np.array(labels, dtype="int64")
    seq_lens = np.array(seq_lens)
    return texts, labels, seq_lens


def do_train():
    paddle.set_device(args.device)
    set_seed(args.seed)

    train_ds, test_ds = load_dataset(
        'msra_ner', splits=('train', 'test'), lazy=False)
    model = FasterErnieForTokenClassification.from_pretrained(
        "ernie-1.0",
        num_classes=len(train_ds.label_list),
        max_seq_len=args.max_seq_length,
        is_split_into_words=True)

    # ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O']
    label_num = len(train_ds.label_list)
    # the label 'O'  index
    no_entity_id = label_num - 1
    # ignore_label is for the label padding
    ignore_label = -100
    trans_func = partial(
        batchify_fn,
        no_entity_id=no_entity_id,
        ignore_label=ignore_label,
        max_seq_len=args.max_seq_length)
    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trans_func,
        return_list=True)
    test_data_loader = DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        collate_fn=trans_func,
        return_list=True)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

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

    criterion = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    metric = ChunkEvaluator(label_list=train_ds.label_list)
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.epochs):
        for step, (texts, labels, seq_lens) in enumerate(
                train_data_loader, start=1):
            texts = to_tensor(texts)
            global_step += 1
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["fused_feedforward", "fused_attention"]):
                logits, preds = model(texts)
                loss = criterion(logits, labels)
            avg_loss = paddle.mean(loss)
            if global_step % 10 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, avg_loss,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            if args.use_amp:
                scaler.scale(avg_loss).backward()
                scaler.minimize(optimizer, avg_loss)
            else:
                avg_loss.backward()
                optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % 500 == 0 or global_step == num_training_steps:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                evaluate(model, criterion, metric, test_data_loader, label_num)
                model.save_pretrained(save_dir)


if __name__ == "__main__":
    do_train()
