"""
Senta Reader
"""

import os
import types
import csv
import numpy as np
from utils import load_vocab
from utils import data_reader

import paddle
import paddle.fluid as fluid

class SentaProcessor(object):
    """
    Processor class for data convertors for senta
    """

    def __init__(self,
                 data_dir,
                 vocab_path,
                 random_seed,
                 max_seq_len):
        self.data_dir = data_dir
        self.vocab = load_vocab(vocab_path)
        self.num_examples = {"train": -1, "dev": -1, "infer": -1}
        np.random.seed(random_seed)
        self.max_seq_len = max_seq_len

    def get_train_examples(self, data_dir, epoch, max_seq_len):
        """
        Load training examples
        """
        return data_reader((self.data_dir + "/train.tsv"), self.vocab, self.num_examples, "train", epoch, max_seq_len)

    def get_dev_examples(self, data_dir, epoch, max_seq_len):
        """
        Load dev examples
        """
        return data_reader((self.data_dir + "/dev.tsv"), self.vocab, self.num_examples, "dev", epoch, max_seq_len)

    def get_test_examples(self, data_dir, epoch, max_seq_len):
        """
        Load test examples
        """
        return data_reader((self.data_dir + "/test.tsv"), self.vocab, self.num_examples, "infer", epoch, max_seq_len)

    def get_labels(self):
        """
        Return Labels
        """
        return ["0", "1"]

    def get_num_examples(self, phase):
        """
        Return num of examples in train, dev, test set
        """
        if phase not in ['train', 'dev', 'infer']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'infer'].")
        return self.num_examples[phase]

    def get_train_progress(self):
        """
        Get train progress
        """
        return self.current_train_example, self.current_train_epoch

    def data_generator(self, batch_size, phase='train', epoch=1, shuffle=True):
        """
        Generate data for train, dev or infer
        """
        if phase == "train":
            return fluid.io.batch(self.get_train_examples(self.data_dir, epoch, self.max_seq_len), batch_size)
            #return self.get_train_examples(self.data_dir, epoch, self.max_seq_len)
        elif phase == "dev":
            return fluid.io.batch(self.get_dev_examples(self.data_dir, epoch, self.max_seq_len), batch_size)
        elif phase == "infer":
            return fluid.io.batch(self.get_test_examples(self.data_dir, epoch, self.max_seq_len), batch_size)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'infer'].")
