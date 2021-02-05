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

from args import parse_args

from data import create_train_loader
from model import Seq2SeqAttnModel, CrossEntropyCriterion

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.metrics import Perplexity


def do_train(args):
    device = paddle.set_device("gpu" if args.use_gpu else "cpu")

    # Define dataloader
    train_loader, vocab_size, pad_id = create_train_loader(args.batch_size)

    model = paddle.Model(
        Seq2SeqAttnModel(vocab_size, args.hidden_size, args.hidden_size,
                         args.num_layers, pad_id))

    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate, parameters=model.parameters())
    ppl_metric = Perplexity()
    model.prepare(optimizer, CrossEntropyCriterion(), ppl_metric)

    print(args)
    model.fit(train_data=train_loader,
              epochs=args.max_epoch,
              eval_freq=1,
              save_freq=1,
              save_dir=args.model_path,
              log_freq=args.log_freq,
              callbacks=[paddle.callbacks.VisualDL('./log')])


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
