import os
import math

from paddle.io import Dataset
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
import paddle.distributed as dist
import numpy as np

__all__ = ['PTBDataset']


class PTBDataset(Dataset):

    DATA_URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
    DATA_PATH = os.path.join('simple-examples', 'data')

    def __init__(self, batch_size, num_steps, mode='train', root=None):
        super(PTBDataset, self).__init__()

        self._get_data(root=root, mode=mode)
        train_data, valid_data, test_data = self.get_ptb_data(self.data_path)
        if mode == 'train':
            raw_data = train_data
        elif mode == 'eval':
            raw_data = valid_data
        else:
            raw_data = test_data
        raw_data = np.asarray(raw_data, dtype="int64")
        self.max_seq_len = len(raw_data) // batch_size
        self.data = raw_data[0:batch_size * self.max_seq_len].reshape(
            (batch_size, self.max_seq_len))
        self.num_steps = num_steps
        self.num_shards = dist.get_world_size()
        index = dist.get_rank()
        self.shard(num_shards=self.num_shards, index=index)

    def _get_data(self, root, mode):
        default_root = os.path.join(DATA_HOME, 'lm')
        self.data_path = os.path.join(default_root,
                                      self.DATA_PATH) if root is None else root
        if not os.path.exists(self.data_path):
            path = get_path_from_url(self.DATA_URL, default_root)
            self.data_path = os.path.join(default_root, self.DATA_PATH)

    def build_vocab(self, filename):
        EOS = "</eos>"
        vocab_dict = {}
        ids = 0
        vocab_dict[EOS] = ids
        ids += 1
        with open(filename, "r") as f:
            for line in f.readlines():
                for w in line.strip().split():
                    if w not in vocab_dict:
                        vocab_dict[w] = ids
                        ids += 1
        self.vocab_size = ids
        return vocab_dict

    def corpus_to_token_ids(self, corpus_path, vocab):
        corpus_ids = []
        with open(corpus_path, "r") as f_corpus:
            for line in f_corpus.readlines():
                tokens = line.strip().split()
                ids = [vocab[w] for w in tokens if w in vocab]

                corpus_ids += ids + [0]  #Add token_id:0 between sentences
        return corpus_ids

    def get_ptb_data(self, data_path=None):

        train_file = os.path.join(data_path, "ptb.train.txt")
        valid_file = os.path.join(data_path, "ptb.valid.txt")
        test_file = os.path.join(data_path, "ptb.test.txt")

        vocab_dict = self.build_vocab(train_file)
        train_ids = self.corpus_to_token_ids(train_file, vocab_dict)
        valid_ids = self.corpus_to_token_ids(valid_file, vocab_dict)
        test_ids = self.corpus_to_token_ids(test_file, vocab_dict)

        return train_ids, valid_ids, test_ids

    def shard(self, num_shards, index):
        num_samples = int(math.floor(len(self.data[0]) * 1.0 / num_shards))
        sharded_data = self.data[:, index * num_samples:(index + 1) *
                                 num_samples]
        self.data = sharded_data

    def __getitem__(self, index):
        x = np.copy(self.data[:, index * self.num_steps:(index + 1) *
                              self.num_steps])
        y = np.copy(self.data[:, index * self.num_steps + 1:(index + 1) *
                              self.num_steps + 1])
        return (x, y)

    def __len__(self):
        return ((self.max_seq_len - 1) // self.num_steps) // self.num_shards
