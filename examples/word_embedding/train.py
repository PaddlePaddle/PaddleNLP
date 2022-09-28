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
import os.path as osp

import paddle
import paddle.nn as nn
import paddlenlp
from paddlenlp.utils.downloader import get_path_from_url
from paddlenlp.embeddings import TokenEmbedding
from paddlenlp.data import JiebaTokenizer, Vocab, Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset

import data

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="Number of epoches for training.")
parser.add_argument("--device", type=str, default="gpu", help="Select cpu, gpu, xpu devices to train model.")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='./checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--use_token_embedding", type=eval, default=True, help="Whether use pretrained embedding")
parser.add_argument("--embedding_name", type=str, default="w2v.baidu_encyclopedia.target.word-word.dim300", help="The name of pretrained embedding")
parser.add_argument("--vdl_dir", type=str, default="vdl_dir/", help="VisualDL log directory")
args = parser.parse_args()
# yapf: enable

WORD_DICT_URL = "https://bj.bcebos.com/paddlenlp/data/dict.txt"


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      pad_token_id=0):
    """
    Creats dataloader.
    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.
    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn, lazy=True)

    shuffle = True if mode == 'train' else False
    sampler = paddle.io.BatchSampler(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=vocab.get('[PAD]', 0)),  # input_ids
        Stack(dtype="int32"),  # seq len
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    dataloader = paddle.io.DataLoader(dataset,
                                      batch_sampler=sampler,
                                      return_list=True,
                                      collate_fn=batchify_fn)
    return dataloader


class BoWModel(nn.Layer):
    """
    This class implements the Bag of Words Classification Network model to classify texts.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `BoWEncoder`.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).
    Args:
        vocab_size (obj:`int`): The vocabulary size.
        emb_dim (obj:`int`, optional, defaults to 300):  The embedding dimension.
        hidden_size (obj:`int`, optional, defaults to 128): The first full-connected layer hidden size.
        fc_hidden_size (obj:`int`, optional, defaults to 96): The second full-connected layer hidden size.
        num_classes (obj:`int`): All the labels that the data has.
    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 vocab_path,
                 emb_dim=300,
                 hidden_size=128,
                 fc_hidden_size=96,
                 use_token_embedding=True):
        super().__init__()
        if use_token_embedding:
            self.embedder = TokenEmbedding(args.embedding_name,
                                           extended_vocab_path=vocab_path)
            emb_dim = self.embedder.embedding_dim
        else:
            padding_idx = vocab_size - 1
            self.embedder = nn.Embedding(vocab_size,
                                         emb_dim,
                                         padding_idx=padding_idx)
        self.bow_encoder = paddlenlp.seq2vec.BoWEncoder(emb_dim)
        self.fc1 = nn.Linear(self.bow_encoder.get_output_dim(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_hidden_size)
        self.dropout = nn.Dropout(p=0.3, axis=1)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, embedding_dim)
        summed = self.bow_encoder(embedded_text)
        summed = self.dropout(summed)
        encoded_text = paddle.tanh(summed)

        # Shape: (batch_size, hidden_size)
        fc1_out = paddle.tanh(self.fc1(encoded_text))
        # Shape: (batch_size, fc_hidden_size)
        fc2_out = paddle.tanh(self.fc2(fc1_out))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc2_out)
        return logits


if __name__ == '__main__':
    assert args.device in [
        "cpu", "gpu", "xpu"
    ], "Invalid device! Available device should be cpu, gpu, or xpu."
    paddle.set_device(args.device)

    # Loads vocab.
    vocab_path = "./dict.txt"
    if not os.path.exists(vocab_path):
        # download in current directory
        get_path_from_url(WORD_DICT_URL, "./")
    vocab = data.load_vocab(vocab_path)

    if '[PAD]' not in vocab:
        vocab['[PAD]'] = len(vocab)
    # Loads dataset.
    train_ds, dev_ds = load_dataset("chnsenticorp", splits=["train", "dev"])

    # Constructs the newtork.
    model = BoWModel(vocab_size=len(vocab),
                     num_classes=len(train_ds.label_list),
                     vocab_path=vocab_path,
                     use_token_embedding=args.use_token_embedding)
    if args.use_token_embedding:
        vocab = model.embedder.vocab
        data.set_tokenizer(vocab)
        vocab = vocab.token_to_idx
    else:
        v = Vocab.from_dict(vocab, unk_token="[UNK]", pad_token="[PAD]")
        data.set_tokenizer(v)
    model = paddle.Model(model)

    # Reads data and generates mini-batches.
    trans_fn = partial(data.convert_example,
                       vocab=vocab,
                       unk_token_id=vocab['[UNK]'],
                       is_test=False)
    train_loader = create_dataloader(train_ds,
                                     trans_fn=trans_fn,
                                     batch_size=args.batch_size,
                                     mode='train',
                                     pad_token_id=vocab['[PAD]'])
    dev_loader = create_dataloader(dev_ds,
                                   trans_fn=trans_fn,
                                   batch_size=args.batch_size,
                                   mode='validation',
                                   pad_token_id=vocab['[PAD]'])

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=args.lr)

    # Defines loss and metric.
    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    model.prepare(optimizer, criterion, metric)

    # Loads pre-trained parameters.
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    # Starts training and evaluating.
    log_dir = 'use_normal_embedding'
    if args.use_token_embedding:
        log_dir = 'use_token_embedding'
    log_dir = osp.join(args.vdl_dir, log_dir)
    callback = paddle.callbacks.VisualDL(log_dir=log_dir)
    model.fit(train_loader,
              dev_loader,
              epochs=args.epochs,
              save_dir=args.save_dir,
              callbacks=callback)
