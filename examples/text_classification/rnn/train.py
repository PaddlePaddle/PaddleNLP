# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import ChnSentiCorp

from utils import convert_example

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument('--use_gpu', type=eval, default=False, help="Whether use GPU for training, input should be True or False")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./senta_word_dict.txt", help="The directory to dataset.")
parser.add_argument('--network', type=str, default="bilstm", help="Which network you would like to choose bow, lstm, bilstm, gru, bigru, rnn, birnn, bilstm_attn and textcnn?")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed=1000):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      use_gpu=False,
                      batchify_fn=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        use_gpu(obj:`bool`, optional, defaults to obj:`False`): Whether to use gpu to run.
        batchify_fn(obj:`callable`, optional, defaults to `None`): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`).

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.apply(trans_fn, lazy=True)

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        shuffle = True if mode == 'train' else False
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_sampler=sampler,
        return_list=True,
        collate_fn=batchify_fn)
    return dataloader


if __name__ == "__main__":
    set_seed()
    paddle.set_device('gpu') if args.use_gpu else paddle.set_device('cpu')

    # Loads vocab.
    if not os.path.exists(args.vocab_path):
        raise RuntimeError('The vocab_path  can not be found in the path %s' %
                           args.vocab_path)

    vocab = Vocab.load_vocabulary(
        args.vocab_path, unk_token='[UNK]', pad_token='[PAD]')
    # Loads dataset.
    train_ds, dev_ds, test_ds = ChnSentiCorp.get_datasets(
        ['train', 'dev', 'test'])

    # Constructs the newtork.
    label_list = train_ds.get_labels()
    model = ppnlp.models.Senta(
        network=args.network,
        vocab_size=len(vocab),
        num_classes=len(label_list))
    model = paddle.Model(model)

    # Reads data and generates mini-batches.
    tokenizer = JiebaTokenizer(vocab)
    trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    train_loader = create_dataloader(
        train_ds,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='train',
        use_gpu=args.use_gpu,
        batchify_fn=batchify_fn)
    dev_loader = create_dataloader(
        dev_ds,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='validation',
        use_gpu=args.use_gpu,
        batchify_fn=batchify_fn)
    test_loader = create_dataloader(
        test_ds,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='test',
        use_gpu=args.use_gpu,
        batchify_fn=batchify_fn)

    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=args.lr)

    # Defines loss and metric.
    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    model.prepare(optimizer, criterion, metric)

    # Loads pre-trained parameters.
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    # Starts training and evaluating.
    callback = paddle.callbacks.ProgBarLogger(log_freq=10, verbose=3)
    model.fit(train_loader,
              dev_loader,
              epochs=args.epochs,
              save_dir=args.save_dir,
              callbacks=callback)

    # Finally tests model.
    results = model.evaluate(test_loader)
    print("Finally test acc: %.5f" % results['acc'])
