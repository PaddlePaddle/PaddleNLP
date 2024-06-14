# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from args import parse_args
from model import CrossEntropyLossForLm, RnnLm, UpdateModel
from reader import create_data_loader

from paddlenlp.metrics import Perplexity

paddle.seed(102)


def train(args):
    paddle.set_device(args.device)
    data_path = args.data_path
    train_loader, valid_loader, test_loader, vocab_size = create_data_loader(
        batch_size=args.batch_size, num_steps=args.num_steps, data_path=data_path
    )

    network = RnnLm(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        init_scale=args.init_scale,
        dropout=args.dropout,
    )
    gloabl_norm_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)
    cross_entropy = CrossEntropyLossForLm()
    ppl_metric = Perplexity()
    callback = UpdateModel()
    scheduler = paddle.callbacks.LRScheduler(by_step=False, by_epoch=True)
    model = paddle.Model(network)

    learning_rate = paddle.optimizer.lr.LambdaDecay(
        learning_rate=args.base_lr,
        lr_lambda=lambda x: args.lr_decay ** max(x + 1 - args.epoch_start_decay, 0.0),
        verbose=True,
    )
    optimizer = paddle.optimizer.SGD(
        learning_rate=learning_rate, parameters=model.parameters(), grad_clip=gloabl_norm_clip
    )

    model.prepare(optimizer=optimizer, loss=cross_entropy, metrics=ppl_metric)

    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    benchmark_logger = paddle.callbacks.ProgBarLogger(log_freq=(len(train_loader) // 10), verbose=3)
    model.fit(
        train_data=train_loader,
        eval_data=valid_loader,
        epochs=args.max_epoch,
        shuffle=False,
        callbacks=[callback, scheduler, benchmark_logger],
    )

    model.save(path="checkpoint/test")  # save for training

    print("Start to evaluate on test dataset...")
    model.evaluate(test_loader, log_freq=len(test_loader))


if __name__ == "__main__":
    args = parse_args()
    train(args)
