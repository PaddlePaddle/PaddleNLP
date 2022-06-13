#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import time
import os
import numpy as np
import random
from functools import partial

from paddlenlp.data import Stack, Tuple, Pad, Dict
import paddle
from paddle.nn import functional as F
from paddle.nn.layer import CrossEntropyLoss
from paddle.io import DataLoader
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import ChineseBertForSequenceClassification
from paddlenlp.transformers import ChineseBertTokenizer
from paddlenlp.datasets import load_dataset

from utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir",
    default='./outputs/chn',
    type=str,
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    "--max_seq_length",
    default=512,
    type=int,
    help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded."
)
parser.add_argument("--batch_size",
                    default=8,
                    type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate",
                    default=2e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay",
                    default=0.0001,
                    type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--epochs",
                    default=10,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt",
                    type=str,
                    default=None,
                    help="The path of checkpoint to be loaded.")
parser.add_argument("--seed",
                    type=int,
                    default=2333,
                    help="random seed for initialization")
parser.add_argument("--device",
                    choices=["cpu", "gpu", "xpu"],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument("--data_path",
                    type=str,
                    default="./data",
                    help="The path of datasets to be loaded")
parser.add_argument("--adam_epsilon",
                    default=1e-8,
                    type=float,
                    help="Epsilon for Adam optimizer.")
args = parser.parse_args()

paddle.set_device(args.device)
set_seed(args.seed)

data_dir = args.data_path
train_path = os.path.join(data_dir, "train.tsv")
dev_path = os.path.join(data_dir, "dev.tsv")
test_path = os.path.join(data_dir, "test.tsv")

from utils import load_ds

train_ds, dev_ds, test_ds = load_ds(datafiles=[train_path, dev_path, test_path])

model = ChineseBertForSequenceClassification.from_pretrained(
    "ChineseBERT-large", num_classes=2)
tokenizer = ChineseBertTokenizer.from_pretrained("ChineseBERT-large")

# model = paddle.DataParallel(model)


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    # The original data is processed into a format that can be read in by the model,
    # enocded_ Inputs is a dict that contains inputs_ids、token_type_ids、etc.
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)

    # input_ids：After the text is segmented into tokens, the corresponding token id in the vocabulary.
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids：Does the current token belong to sentence 1 or sentence 2, that is, the segment ids.

    token_type_ids = encoded_inputs["token_type_ids"]
    pinyin_ids = encoded_inputs["pinyin_ids"]

    label = np.array([example["label"]], dtype="int64")
    return input_ids, pinyin_ids, label


# Process the data into a data format that the model can read in.
trans_func = partial(convert_example,
                     tokenizer=tokenizer,
                     max_seq_length=args.max_seq_length)

# Form data into batch data, such as padding text sequences of different lengths into the maximum length of batch data,
# and stack each data label together
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    # Pad(axis=0, pad_val=tokenizer.pad_token_type_id), # token_type_ids
    Pad(axis=0, pad_val=0),  # pinyin_ids
    Stack()  # labels
): [data for data in fn(samples)]

from utils import create_dataloader

train_data_loader = create_dataloader(train_ds,
                                      mode='train',
                                      batch_size=args.batch_size,
                                      batchify_fn=batchify_fn,
                                      trans_fn=trans_func)
dev_data_loader = create_dataloader(dev_ds,
                                    mode='dev',
                                    batch_size=args.batch_size,
                                    batchify_fn=batchify_fn,
                                    trans_fn=trans_func)
test_data_loader = create_dataloader(test_ds,
                                     mode='test',
                                     batch_size=args.batch_size,
                                     batchify_fn=batchify_fn,
                                     trans_fn=trans_func)

from utils import evaluate

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
    beta1=0.9,
    beta2=0.98,
    learning_rate=lr_scheduler,
    epsilon=args.adam_epsilon,
    parameters=model.parameters(),
    weight_decay=args.weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params)

# cross-entropy cost function
criterion = paddle.nn.loss.CrossEntropyLoss()
# accuracy metric
metric = paddle.metric.Accuracy()
print(args)
# train
global_step = 0
tic_train = time.time()
for epoch in range(1, args.epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, pinyin_ids, labels = batch
        batch_size, length = input_ids.shape
        pinyin_ids = paddle.reshape(pinyin_ids, [batch_size, length, 8])
        logits = model(input_ids, pinyin_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)

        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc, 10 /
                   (time.time() - tic_train)))
            tic_train = time.time()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            dev_acc = evaluate(model, criterion, metric, dev_data_loader)
            test_acc = evaluate(model, criterion, metric, test_data_loader)
            if test_acc >= 0.959:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
