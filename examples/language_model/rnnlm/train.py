import os
import sys
sys.path.append('../../..')

import paddle
from paddle.io import DataLoader
from model import RnnLm, CrossEntropyLossForLm, UpdateModel
from args import parse_args
from paddlenlp.datasets import PTBDataset
from paddlenlp.metrics import Perplexity

paddle.seed(102)


def create_data_loader(batch_size, num_steps, data_path):
    train_ds = PTBDataset(batch_size, num_steps, 'train')
    valid_ds = PTBDataset(batch_size, num_steps, 'eval')
    test_ds = PTBDataset(batch_size, num_steps, 'test')

    train_loader = DataLoader(train_ds, return_list=True, batch_size=None)
    valid_loader = DataLoader(valid_ds, return_list=True, batch_size=None)
    test_loader = DataLoader(test_ds, return_list=True, batch_size=None)
    return train_loader, valid_loader, test_loader


def train(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    data_path = args.data_path
    train_loader, valid_loader, test_loader = create_data_loader(
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        data_path=data_path)

    network = RnnLm(
        vocab_size=train_loader.dataset.vocab_size,
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
        lr_lambda=lambda x: args.lr_decay**max(x + 1 - args.epoch_start_decay, 0.0),
        verbose=True)
    optimizer = paddle.optimizer.SGD(learning_rate=learning_rate,
                                     parameters=model.parameters(),
                                     grad_clip=gloabl_norm_clip)

    model.prepare(optimizer=optimizer, loss=cross_entropy, metrics=ppl_metric)

    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    model.fit(train_data=train_loader,
              eval_data=valid_loader,
              epochs=args.max_epoch,
              shuffle=False,
              callbacks=[callback, scheduler],
              log_freq=max(1, len(train_loader) // 10))

    model.save(path='checkpoint/test')  # save for training

    print('Start to evaluate on test dataset...')
    model.evaluate(test_loader, log_freq=len(test_loader))


if __name__ == '__main__':
    args = parse_args()
    if args.n_gpu > 1:
        paddle.distributed.spawn(train, args=(args, ), nprocs=args.n_gpu)
    else:
        train(args)
