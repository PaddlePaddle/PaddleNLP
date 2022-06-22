#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import numpy as np
import pandas as pd

from data import CovidDataset
from model import TCNNetwork

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--data_path", type=str, default="time_series_covid19_confirmed_global.csv", help="The data path.")
parser.add_argument("--seq_length", type=int, default=8, help="The time series length.")
parser.add_argument("--test_data_size", type=int, default=30, help="The data will be split to a train set and a test set. test_data_size determines the test set size.")
parser.add_argument("--batch_size", type=int, default=8, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--epochs", type=int, default=100, help="train iteration number.")
parser.add_argument("--lr", type=float, default=0.001, help="The learning rate.")
parser.add_argument("--use_gpu", action='store_true', default=False, help="If set, use GPU for training.")
parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to init model.")
parser.add_argument("--model_save_dir", type=str, default="save_dir", help="The model will be saved in this path.")
args = parser.parse_args()
# yapf: enable


def train():
    if args.use_gpu:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    train_dataset = CovidDataset(args.data_path,
                                 args.test_data_size,
                                 args.seq_length,
                                 mode="train")

    network = TCNNetwork(input_size=1)

    model = paddle.Model(network)

    optimizer = paddle.optimizer.Adam(learning_rate=args.lr,
                                      parameters=model.parameters())

    loss = paddle.nn.MSELoss(reduction='sum')

    model.prepare(optimizer, loss)

    if args.init_checkpoint:
        model.load(args.init_checkpoint)

    model.fit(train_dataset,
              batch_size=32,
              drop_last=True,
              epochs=args.epochs,
              save_dir=args.model_save_dir,
              save_freq=10,
              verbose=1)


if __name__ == "__main__":
    print(args)
    train()
