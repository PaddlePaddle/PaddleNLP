import os
import sys
import paddle
import numpy as np

from model import RnnLm, CrossEntropyLossForLm, UpdateModel
from args import parse_args
from reader import create_data_loader

from paddlenlp.metrics import Perplexity

paddle.seed(102)


def train(args):
    paddle.set_device(args.device)
    data_path = args.data_path
    train_loader, valid_loader, test_loader, vocab_size = create_data_loader(
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        data_path=data_path)

    network = RnnLm(vocab_size=vocab_size,
                    hidden_size=args.hidden_size,
                    batch_size=args.batch_size,
                    num_layers=args.num_layers,
                    init_scale=args.init_scale,
                    dropout=args.dropout)
    gloabl_norm_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)
    cross_entropy = CrossEntropyLossForLm()
    ppl_metric = Perplexity()
    callback = UpdateModel()
    scheduler = paddle.callbacks.LRScheduler(by_step=False, by_epoch=True)
    model = paddle.Model(network)

    learning_rate = paddle.optimizer.lr.LambdaDecay(
        learning_rate=args.base_lr,
        lr_lambda=lambda x: args.lr_decay**max(x + 1 - args.epoch_start_decay,
                                               0.0),
        verbose=True)
    optimizer = paddle.optimizer.SGD(learning_rate=learning_rate,
                                     parameters=model.parameters(),
                                     grad_clip=gloabl_norm_clip)

    model.prepare(optimizer=optimizer, loss=cross_entropy, metrics=ppl_metric)

    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    benchmark_logger = paddle.callbacks.ProgBarLogger(
        log_freq=(len(train_loader) // 10), verbose=3)
    model.fit(train_data=train_loader,
              eval_data=valid_loader,
              epochs=args.max_epoch,
              shuffle=False,
              callbacks=[callback, scheduler, benchmark_logger])

    model.save(path='checkpoint/test')  # save for training

    print('Start to evaluate on test dataset...')
    model.evaluate(test_loader, log_freq=len(test_loader))


if __name__ == '__main__':
    args = parse_args()
    train(args)
