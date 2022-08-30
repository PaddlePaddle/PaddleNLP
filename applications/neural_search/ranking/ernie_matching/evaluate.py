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

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup, AutoModel, AutoTokenizer

from data import create_dataloader, gen_pair
from data import convert_pairwise_example as convert_example
from model import PairwiseMatching
import pandas as pd
from tqdm import tqdm

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--margin", default=0.1, type=float, help="Margin for pos_score and neg_score.")
parser.add_argument("--test_file", type=str, required=True, help="The full path of test file")

parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--eval_step", default=200, type=int, help="Step interval for evaluation.")
parser.add_argument('--save_step', default=10000, type=int, help="Step interval for saving checkpoint.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


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

        pos_probs = model.predict(input_ids=input_ids,
                                  token_type_ids=token_type_ids)

        neg_probs = 1.0 - pos_probs

        preds = np.concatenate((neg_probs, pos_probs), axis=1)
        metric.update(preds=preds, labels=labels)

    print("eval_{} auc:{:.3}".format(phase, metric.accumulate()))
    metric.reset()
    model.train()


# 构建读取函数，读取原始数据
def read(src_path, is_predict=False):
    data = pd.read_csv(src_path, sep='\t')
    for index, row in tqdm(data.iterrows()):
        query = row['query']
        title = row['title']
        neg_title = row['neg_title']
        yield {'query': query, 'title': title, 'neg_title': neg_title}


def read_test(src_path, is_predict=False):
    data = pd.read_csv(src_path, sep='\t')
    for index, row in tqdm(data.iterrows()):
        query = row['query']
        title = row['title']
        label = row['label']
        yield {'query': query, 'title': title, 'label': label}


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    dev_ds = load_dataset(read_test, src_path=args.test_file, lazy=False)
    print(dev_ds[0])

    pretrained_model = AutoModel.from_pretrained('ernie-3.0-medium-zh')
    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-medium-zh')

    trans_func_eval = partial(convert_example,
                              tokenizer=tokenizer,
                              max_seq_length=args.max_seq_length,
                              phase="eval")

    batchify_fn_eval = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"
            ),  # pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"
            ),  # pair_segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    dev_data_loader = create_dataloader(dev_ds,
                                        mode='dev',
                                        batch_size=args.batch_size,
                                        batchify_fn=batchify_fn_eval,
                                        trans_fn=trans_func_eval)

    model = PairwiseMatching(pretrained_model, margin=args.margin)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    metric = paddle.metric.Auc()
    evaluate(model, metric, dev_data_loader, "dev")


if __name__ == "__main__":
    do_train()
