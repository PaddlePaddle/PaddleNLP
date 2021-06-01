import os
import sys
import paddle
import numpy as np

from model import RnnLm, CrossEntropyLossForLm, UpdateModel
from args import parse_args

from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import Perplexity
from paddlenlp.data import Vocab

paddle.seed(102)


def create_data_loader(batch_size, num_steps, data_path):
    train_ds, valid_ds, test_ds = load_dataset(
        'ptb', splits=('train', 'valid', 'test'))

    train_examples = [
        train_ds[i]['sentence'].split() for i in range(len(train_ds))
    ]
    vocab = Vocab.build_vocab(train_examples, eos_token='</eos>')

    # Because the sentences in PTB dataset might be consecutive, we need to concatenate 
    # all texts from our dataset and fold them into chunks while the number of rows is 
    # equal to batch size. For example:
    #
    #   Sentence1: we're talking about years ago before anyone heard of asbestos having 
    #              any questionable properties. 
    #   Sentence2: there is no asbestos in our products now.
    #   Batch_size: 5
    #   Grouped_text: [["we're", "talking", "about", "years"],
    #                  ["ago", "before", "anyone", "heard"],
    #                  ["of", "asbestos", "having", "any"],
    #                  ["questionable", "properties", "there", "is"],
    #                  ["no", "asbestos", "in", "our"]] 
    #
    def group_texts(examples):
        concat_examples = []
        for example in examples:
            concat_examples += example['sentence'].split() + ['</eos>']

        concat_examples = vocab.to_indices(concat_examples)

        max_seq_len = len(concat_examples) // batch_size
        reshaped_examples = np.asarray(
            concat_examples[0:batch_size * max_seq_len], dtype='int64').reshape(
                (batch_size, max_seq_len))
        encoded_examples = []
        for i in range(max_seq_len // num_steps):
            encoded_examples.append(
                (np.copy(reshaped_examples[:, i * num_steps:(i + 1) *
                                           num_steps]),
                 np.copy(reshaped_examples[:, i * num_steps + 1:(i + 1) *
                                           num_steps + 1])))

        return encoded_examples

    train_ds.map(group_texts, batched=True)
    valid_ds.map(group_texts, batched=True)
    test_ds.map(group_texts, batched=True)

    train_loader = paddle.io.DataLoader(
        train_ds, return_list=True, batch_size=None)
    valid_loader = paddle.io.DataLoader(
        valid_ds, return_list=True, batch_size=None)
    test_loader = paddle.io.DataLoader(
        test_ds, return_list=True, batch_size=None)
    return train_loader, valid_loader, test_loader, len(vocab)


def train(args):
    paddle.set_device(args.device)
    data_path = args.data_path
    train_loader, valid_loader, test_loader, vocab_size = create_data_loader(
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        data_path=data_path)

    network = RnnLm(
        vocab_size=vocab_size,
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
