# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import math
import numpy as np

import paddle
from paddle.io import Dataset
from paddlenlp.data import Vocab
from utils import kmeans, pad_sequence


def build_vocab(corpus, tokenizer, encoding_model, feat):
    """
    Build vocabs use the api of paddlenlp.data.Vocab.build_vocab(), 
    Using token_to_idx to specifies the mapping relationship between 
    tokens and indices to be used.

    Args:
        Corpus(obj:`list[list[str]]`): The training corpus which contains 
            list of input words, features and relations.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from 
            :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. If the encoding model is lstm,
            tokenizer is None.
        encoding_model(obj:`str`): The encoder used for embedding.
        feat(obj:`str`): The features used for model inputs. If the encoding
            model is lstm, feat can be `pos` or `char`, otherwise the feat is None.

    Returns:
        word_vocab(obj:`Vocab`): Word vocab.
        feat_vocab(obj:`Vocab`): Feature vocab.
        rel_vocab(obj:`Vocab`): Relation vocab.
    """
    word_examples, feat_examples, rel_examples = corpus

    # Build word vocab and feature vocab
    if encoding_model == "lstm":
        # Using token_to_idx to specifies the mapping
        # relationship between tokens and indices
        word_vocab = Vocab.build_vocab(
            word_examples,
            min_freq=2,
            token_to_idx={
                "[PAD]": 0,
                "[UNK]": 1,
                "[BOS]": 2,
                "[EOS]": 3
            },
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )
        if feat == "pos":
            feat_vocab = Vocab.build_vocab(
                feat_examples,
                token_to_idx={
                    "[BOS]": 0,
                    "[EOS]": 1
                },
                bos_token="[BOS]",
                eos_token="[EOS]",
            )
        else:
            feat_vocab = Vocab.build_vocab(
                feat_examples,
                token_to_idx={
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "[BOS]": 2,
                    "[EOS]": 3
                },
                unk_token="[UNK]",
                pad_token="[PAD]",
                bos_token="[BOS]",
                eos_token="[EOS]",
            )
    else:
        word_vocab = tokenizer.vocab
        feat_vocab = None

    # Build relation vocab
    rel_vocab = Vocab.build_vocab(
        rel_examples,
        token_to_idx={
            "[BOS]": 0,
            "[EOS]": 1,
            "[UNK]": 2
        },
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    return word_vocab, feat_vocab, rel_vocab


def load_vocab(vocab_dir):
    """load vocabs"""
    word_vocab = Vocab.from_json(os.path.join(vocab_dir, "word_vocab.json"))
    rel_vocab = Vocab.from_json(os.path.join(vocab_dir, "rel_vocab.json"))
    feat_vocab_path = os.path.join(vocab_dir, "feat_vocab.json")
    if os.path.exists(feat_vocab_path):
        feat_vocab = Vocab.from_json(os.path.join(feat_vocab_path))
    else:
        feat_vocab = None
    return word_vocab, feat_vocab, rel_vocab


def convert_example(example,
                    vocabs,
                    encoding_model='ernie-3.0-medium-zh',
                    feat=None,
                    mode='train',
                    fix_len=20):
    """Builds model inputs for dependency parsing task."""
    word_vocab, feat_vocab, rel_vocab = vocabs
    if encoding_model == "lstm":
        word_bos_index = word_vocab.to_indices("[BOS]")
        word_eos_index = word_vocab.to_indices("[EOS]")
    else:
        word_bos_index = word_vocab.to_indices("[CLS]")
        word_eos_index = word_vocab.to_indices("[SEP]")

    if feat_vocab:
        feat_bos_index = feat_vocab.to_indices("[BOS]")
        feat_eos_index = feat_vocab.to_indices("[EOS]")

    arc_bos_index, arc_eos_index = 0, 1

    rel_bos_index = rel_vocab.to_indices("[BOS]")
    rel_eos_index = rel_vocab.to_indices("[EOS]")

    if mode != "test":
        arcs = list(example["HEAD"])
        arcs = [arc_bos_index] + arcs + [arc_eos_index]
        arcs = np.array(arcs, dtype=int)

        rels = rel_vocab.to_indices(example["DEPREL"])
        rels = [rel_bos_index] + rels + [rel_eos_index]
        rels = np.array(rels, dtype=int)

    if encoding_model == "lstm":
        words = word_vocab.to_indices(example["FORM"])
        words = [word_bos_index] + words + [word_eos_index]
        words = np.array(words, dtype=int)

        if feat == "pos":
            feats = feat_vocab.to_indices(example["CPOS"])
            feats = [feat_bos_index] + feats + [feat_eos_index]
            feats = np.array(feats, dtype=int)
        else:
            feats = [[feat_vocab.to_indices(token) for token in word]
                     for word in example["FORM"]]
            feats = [[feat_bos_index]] + feats + [[feat_eos_index]]
            feats = pad_sequence(
                [np.array(ids[:fix_len], dtype=int) for ids in feats],
                fix_len=fix_len)
        if mode == "test":
            return words, feats
        return words, feats, arcs, rels
    else:
        words = [[word_vocab.to_indices(char) for char in word]
                 for word in example["FORM"]]
        words = [[word_bos_index]] + words + [[word_eos_index]]
        words = pad_sequence(
            [np.array(ids[:fix_len], dtype=int) for ids in words],
            fix_len=fix_len)
        if mode == "test":
            return [words]
        return words, arcs, rels


def create_dataloader(dataset,
                      batch_size,
                      mode="train",
                      n_buckets=None,
                      trans_fn=None):
    """
    Create dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will 
            shuffle the dataset randomly.
        n_buckets(obj:`int`, optional, defaults to `None`): If n_buckets is not None, it will devide 
            the dataset into n_buckets according to the sequence lengths.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a 
            data sample to input ids, etc.
    """
    if n_buckets:
        word_examples = [seq["FORM"] for seq in dataset]
        lengths = [len(i) + 1 for i in word_examples]
        buckets = dict(zip(*kmeans(lengths, n_buckets)))
    else:
        buckets = None
    if trans_fn:
        dataset = dataset.map(trans_fn)

    if n_buckets:
        if mode == "train":
            batch_sampler = BucketsSampler(
                buckets=buckets,
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            batch_sampler = BucketsSampler(
                buckets=buckets,
                batch_size=batch_size,
                shuffle=False,
            )
    else:
        batch_sampler = SequentialSampler(
            batch_size=batch_size,
            corpus_length=len(dataset),
        )

    # Subclass of `paddle.io.Dataset`
    dataset = Batchify(dataset, batch_sampler)

    # According to the api of `paddle.io.DataLoader` set `batch_size`
    # and `batch_sampler` to `None` to disable batchify dataset automatically
    data_loader = paddle.io.DataLoader(dataset=dataset,
                                       batch_sampler=None,
                                       batch_size=None,
                                       return_list=True)
    return data_loader, buckets


class Batchify(Dataset):

    def __init__(self, dataset, batch_sampler):

        self.batches = []
        for batch_sample_id in batch_sampler:
            batch = []
            raw_batch = self._collate_fn(
                [dataset[sample_id] for sample_id in batch_sample_id])
            for data in raw_batch:
                if isinstance(data[0], np.ndarray):
                    data = pad_sequence(data)
                batch.append(data)
            self.batches.append(batch)

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)

    def _collate_fn(self, batch):
        """Return batch samples"""
        return (raw for raw in zip(*batch))


class BucketsSampler(object):
    """BucketsSampler"""

    def __init__(self, buckets, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[(size, bucket)
                                         for size, bucket in buckets.items()])
        # The number of chunks in each bucket, which is clipped by range [1, len(bucket)]
        self.chunks = []
        for size, bucket in zip(self.sizes, self.buckets):
            max_ch = max(math.ceil(size * len(bucket) / batch_size), 1)
            chunk = min(len(bucket), int(max_ch))
            self.chunks.append(chunk)

    def __iter__(self):
        """Returns an iterator, randomly or sequentially returns a batch id"""
        range_fn = np.random.permutation if self.shuffle else np.arange
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1
                           for j in range(self.chunks[i])]
            for batch in np.split(range_fn(len(self.buckets[i])),
                                  np.cumsum(split_sizes)):
                if len(batch):
                    yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        """Returns the number of batches"""
        return sum(self.chunks)


class SequentialSampler(object):
    """SequentialSampler"""

    def __init__(self, batch_size, corpus_length):
        self.batch_size = batch_size
        self.corpus_length = corpus_length

    def __iter__(self):
        """iter"""
        batch = []
        for i in range(self.corpus_length):
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        else:
            if len(batch):
                yield batch
