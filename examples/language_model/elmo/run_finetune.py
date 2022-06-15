import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
import paddle.distributed as dist

import os
import re
import argparse
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split

from dataset import load_vocab
from elmo import get_elmo_layer


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="./sentence-polarity-dataset-v1/", help="Specify the data dir.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--logging_step", type=int, default=10, help="The frequency, in number of steps, the training logs are printed. (default: %(default)d)")
    parser.add_argument("--epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--dropout", type=float, default=0.5, help="The dropout rate.")
    parser.add_argument("--lr", type=float, default=0.001, help="The initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="The weight decay for optimizer.")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed.")
    parser.add_argument("--max_seq_len", type=int, default=256, help='max grad norm')
    parser.add_argument("--sent_embedding_dim", type=int, default=64, help="The size of sentence embedding.")
    parser.add_argument("--num_classes", type=int, default=2, help="The num of classification classes.")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device for selecting for the training.")

    args = parser.parse_args()
    return args
# yapf: enable


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(
        open(positive_data_file, 'r', encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(
        open(negative_data_file, 'r', encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = list(map(lambda x: x.split(), x_text))
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.array(positive_labels + negative_labels)
    return [x_text, y]


class ELMoBowTextClassification(nn.Layer):

    def __init__(self, params_file, batch_size, sent_embedding_dim, dropout,
                 num_labels):
        super(ELMoBowTextClassification, self).__init__()

        self._elmo = get_elmo_layer(params_file, batch_size, trainable=True)
        word_embedding_dim = self._elmo.embedding_dim
        self._fc1 = nn.Linear(word_embedding_dim, sent_embedding_dim)
        self._fc2 = nn.Linear(sent_embedding_dim, num_labels)
        self._dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        """
        Parameters:
            inputs (Tuple): It is a Tuple contains 2 tensor with shape 
                `[batch_size, max_seq_len, max_characters_per_token]`. 
        """
        mask = paddle.any(inputs[0] > 0, axis=2)
        # [batch_size, 3, max_seq_len, word_embedding_dim]
        elmo_out = self._elmo(inputs)
        # [batch_size, max_seq_len, word_embedding_dim]
        word_emb = self.mix_elmo_outputs(elmo_out)

        # [batch_size, word_embedding_dim]
        sent_emb = self.average_word_embedding(word_emb, mask)

        # [batch_size, sent_embedding_dim]
        dense = self._fc1(sent_emb)
        dense = self._dropout(dense)

        # [batch_size, num_labels]
        out = self._fc2(dense)
        return out

    def mix_elmo_outputs(self, elmo_out):
        """
        Computes a mixture of elmo_out. At present, we simply take the last one.
        Parameters:
            elmo_out (Tensor): It is a Tensor with shape 
                `[batch_size, 3, max_seq_len, word_embedding_dim]`. 
        """
        # [batch_size, max_seq_len, word_embedding_dim]
        word_emb = elmo_out[:, 2, :, :]
        return word_emb

    def average_word_embedding(self, word_emb, mask):
        """
        Parameters:
            word_emb: It is a Tensor with shape `[batch_size, max_seq_len, word_embedding_dim]`.
            mask: It is a Tensor with shape `[batch_size, max_seq_len]`.
        """
        mask = paddle.unsqueeze(mask, axis=-1)
        # [batch_size, 1]
        seq_lens = paddle.sum(paddle.cast(mask, dtype=word_emb.dtype), axis=1)

        # [batch_size, max_seq_len, word_embedding_dim]
        word_emb = word_emb * mask
        # [batch_size, word_embedding_dim]
        sent_emb = paddle.sum(word_emb, axis=1)
        # [batch_size, word_embedding_dim]
        sent_emb = sent_emb / seq_lens
        return sent_emb


class SentencePolarityDatasetV1(Dataset):

    def __init__(self, x, y, vocab, max_seq_len):
        super(SentencePolarityDatasetV1, self).__init__()

        self._text = list(zip(x, y))
        self._vocab = vocab
        self._max_seq_len = max_seq_len
        self._data = self.convert_to_ids()

    def convert_to_ids(self):
        data = []
        for sentence, label in self._text:
            ids, ids_reverse = self._vocab.encode_chars(
                sentence[:self._max_seq_len - 2], split=False)
            data.append([ids, ids_reverse, label])
        return data

    def __getitem__(self, idx):
        ids = np.copy(self._data[idx][0])
        ids_reverse = np.copy(self._data[idx][1])
        label = self._data[idx][2]
        return (ids, ids_reverse, label)

    def __len__(self):
        return len(self._data)


def generate_batch(batch):
    batch_ids, batch_ids_reverse, batch_label = zip(*batch)
    max_len = max([ids.shape[0] for ids in batch_ids])
    new_batch_ids = np.zeros([len(batch_ids), max_len, batch_ids[0].shape[1]],
                             dtype=np.int64)
    new_batch_ids_reverse = np.zeros(
        [len(batch_ids), max_len, batch_ids[0].shape[1]], dtype=np.int64)
    new_batch_label = []
    for i, (ids, ids_reverse,
            label) in enumerate(zip(batch_ids, batch_ids_reverse, batch_label)):
        seq_len = ids.shape[0]
        new_batch_ids[i, :seq_len, :] = ids
        new_batch_ids_reverse[i, :seq_len, :] = ids_reverse
        new_batch_label.append(label)
    return new_batch_ids, new_batch_ids_reverse, new_batch_label


def finetune(args):
    paddle.set_device(args.device)
    if dist.get_world_size() > 1:
        dist.init_parallel_env()

    pos_file = os.path.join(args.data_dir, 'rt-polarity.pos')
    neg_file = os.path.join(args.data_dir, 'rt-polarity.neg')
    x_text, y = load_data_and_labels(pos_file, neg_file)
    x_train, x_test, y_train, y_test = train_test_split(x_text,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=args.seed)

    if not args.init_from_ckpt:
        raise ValueError('`init_from_ckpt` should be set.')
    model = ELMoBowTextClassification(args.init_from_ckpt, args.batch_size,
                                      args.sent_embedding_dim, args.dropout,
                                      args.num_classes)
    if dist.get_world_size() > 1:
        model = paddle.DataParallel(model)
    model.train()

    adam = paddle.optimizer.Adam(parameters=model.parameters(),
                                 learning_rate=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    vocab = load_vocab()

    train_dataset = SentencePolarityDatasetV1(x_train, y_train, vocab,
                                              args.max_seq_len)
    test_dataset = SentencePolarityDatasetV1(x_test, y_test, vocab,
                                             args.max_seq_len)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              return_list=True,
                              shuffle=True,
                              collate_fn=lambda batch: generate_batch(batch))
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             return_list=True,
                             shuffle=False,
                             collate_fn=lambda batch: generate_batch(batch))

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        for step, batch_data in enumerate(train_loader, start=1):
            ids, ids_reverse, label = batch_data

            output = model((ids, ids_reverse))
            loss = criterion(output, label)
            loss.backward()
            adam.step()
            adam.clear_grad()

            if step % args.logging_step == 0:
                print('step {}, loss {}'.format(step, loss.numpy()[0]))

    acc = test(model, test_loader)
    print('\ntest acc {}\n'.format(acc))


@paddle.no_grad()
def test(model, test_loader):
    correct = num = 0
    model.eval()
    for batch_data in test_loader:
        ids, ids_reverse, label = batch_data

        # [batch_size, 2]
        output = model((ids, ids_reverse))

        num += label.shape[0]
        predict = paddle.argmax(output, axis=1)
        label = paddle.cast(label, dtype=predict.dtype)
        correct += paddle.sum(paddle.cast(predict == label,
                                          dtype='int64')).numpy()[0]
    model.train()
    return correct * 1.0 / num


if __name__ == '__main__':
    args = parse_args()
    finetune(args)
