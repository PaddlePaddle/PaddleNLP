# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import numpy as np
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab

from data import create_dataloader, convert_example, read_custom_data
from model import TextCNNModel

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--data_path", type=str, default='./RobotChat', help="The path of datasets to be loaded")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./robot_chat_word_dict.txt", help="The directory to dataset.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed=1000):
    """Sets random seed."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


if __name__ == "__main__":
    paddle.set_device(args.device)
    set_seed()

    # Load vocab.
    if not os.path.exists(args.vocab_path):
        raise RuntimeError('The vocab_path  can not be found in the path %s' %
                           args.vocab_path)

    vocab = Vocab.load_vocabulary(args.vocab_path,
                                  unk_token='[UNK]',
                                  pad_token='[PAD]')

    # Load datasets.
    dataset_names = ['train.tsv', 'dev.tsv', 'test.tsv']
    train_ds, dev_ds, test_ds = [load_dataset(read_custom_data, \
        filename=os.path.join(args.data_path, dataset_name), lazy=False) for dataset_name in dataset_names]

    tokenizer = JiebaTokenizer(vocab)
    trans_fn = partial(convert_example, tokenizer=tokenizer)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),
        Stack(dtype='int64')  # label
    ): [data for data in fn(samples)]
    train_loader = create_dataloader(train_ds,
                                     batch_size=args.batch_size,
                                     mode='train',
                                     batchify_fn=batchify_fn,
                                     trans_fn=trans_fn)
    dev_loader = create_dataloader(dev_ds,
                                   batch_size=args.batch_size,
                                   mode='validation',
                                   batchify_fn=batchify_fn,
                                   trans_fn=trans_fn)
    test_loader = create_dataloader(test_ds,
                                    batch_size=args.batch_size,
                                    mode='test',
                                    batchify_fn=batchify_fn,
                                    trans_fn=trans_fn)

    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    vocab_size = len(vocab)
    num_classes = len(label_map)
    pad_token_id = vocab.to_indices('[PAD]')

    model = TextCNNModel(vocab_size,
                         num_classes,
                         padding_idx=pad_token_id,
                         ngram_filter_sizes=(1, 2, 3))

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    model = paddle.Model(model)

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=args.lr)

    # Define loss and metric.
    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    model.prepare(optimizer, criterion, metric)

    # Start training and evaluating.
    callback = paddle.callbacks.ProgBarLogger(log_freq=10, verbose=3)
    model.fit(train_loader,
              dev_loader,
              epochs=args.epochs,
              save_dir=args.save_dir,
              callbacks=callback)

    # Evaluate on test dataset
    print('Start to evaluate on test dataset...')
    model.evaluate(test_loader, log_freq=len(test_loader))
