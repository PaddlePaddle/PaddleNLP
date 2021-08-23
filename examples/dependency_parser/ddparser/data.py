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

import math
import numpy as np

import paddle
import paddlenlp as ppnlp
from paddlenlp.data import Vocab
from utils import kmeans, pad_sequence


def build_vocab(corpus, tokenizer, encoding_model, feat):
    word_examples, feat_examples, rel_examples = corpus

    # Construct word vocab and feature vocab
    if encoding_model == "lstm":
        word_vocab = Vocab.build_vocab(
            word_examples, 
            min_freq=2, 
            token_to_idx={"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3},
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )
        if feat == "pos":
            feat_vocab = Vocab.build_vocab(
                feat_examples,
                token_to_idx={"[BOS]": 0, "[EOS]": 1, "[UNK]": 2},
                bos_token="[BOS]",
                eos_token="[EOS]",
                unk_token="[UNK]",
            )
        else:
            feat_vocab = Vocab.build_vocab(
                feat_examples,
                token_to_idx={"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3},
                unk_token="[UNK]",
                pad_token="[PAD]",
                bos_token="[BOS]",
                eos_token="[EOS]",
            )
    else:
        word_vocab = tokenizer.vocab
        feat_vocab = None

    # Construct relation vocab
    rel_vocab = Vocab.build_vocab(
        rel_examples,
        token_to_idx={"[BOS]": 0, "[EOS]": 1, "[UNK]": 2},
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )

    return [word_vocab, feat_vocab, rel_vocab]

def convert_example(example, tokenizer, vocabs, encoding_model, feat, fix_len=20):
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

    arcs = list(example["HEAD"])
    arcs = [arc_bos_index] + arcs + [arc_eos_index]
    arcs = np.array(arcs, dtype=np.int64)

    rels = rel_vocab.to_indices(example["DEPREL"])
    rels = [rel_bos_index] + rels + [rel_eos_index]
    rels = np.array(rels, dtype=np.int64)
    if encoding_model == "lstm":
        words = word_vocab.to_indices(example["FORM"])
        words = [word_bos_index] + words + [word_eos_index]
        words = np.array(words, dtype=np.int64)

        if feat == "pos":
            feats = feat_vocab.to_indices(example["CPOS"])
            feats = [feat_bos_index] + feats + [feat_eos_index]
            feats = np.array(feats, dtype=np.int64)
        else:
            feats = [[feat_vocab.to_indices(token) for token in word] 
                for word in example["FORM"]]
            feats = [[feat_bos_index]] + feats + [[feat_eos_index]]
            feats = pad_sequence([np.array(ids[:fix_len], dtype=np.int64)
                for ids in feats], fix_len=fix_len)

        return words, feats, arcs, rels
    else:
        words = [tokenizer(word)["input_ids"][1:-1] for word in example["FORM"]]
        words = [[word_bos_index]] + words + [[word_eos_index]]
        words = pad_sequence([np.array(ids[:fix_len], dtype=np.int64) 
            for ids in words], fix_len=fix_len)

        return words, arcs, rels


def create_dataloader(dataset, 
                      vocabs,
                      batch_size, 
                      mode="train", 
                      n_buckets=15,
                      trans_fn=None): 
    if n_buckets:
        word_examples = [seq["FORM"] for seq in dataset]
        lengths = [len(i) + 1 for i in word_examples]
        buckets = dict(zip(*kmeans(lengths, n_buckets)))

    if trans_fn:
        dataset = dataset.map(trans_fn)

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
    dataloader = paddle.io.DataLoader.from_generator(
        capacity=10, 
        return_list=True, 
        use_multiprocess=True,
    )
    dataloader.set_batch_generator(
        generator_creator(dataset, batch_sampler))
  
    return dataloader, buckets
    

def read_predict_data(filename):
    """Reads data."""
    start = 0
    with open(filename, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            if not line.startswith(" "):
                if not line.startswith('#') and (len(line) == 1 or line.split()[0].isdigit()):
                    lines.append(line.strip())
            else:
                lines.append("")

    for i, line in enumerate(lines):
        if not line:
            values = list(zip(*[j.split('\t') for j in lines[start:i]]))
            if len(values) == 10:
                # CONLL-X format (NLPCC13_EVSAM05_HIT style)
                ID, FORM, LEMMA, CPOS, POS, FEATS, HEAD, DEPREL, PHEAD, PDEPREL = values
            else:
                # CONLL-X format (NLPCC13_EVSAM05_THU style)
                ID, FORM, LEMMA, CPOS, POS, FEATS, HEAD, DEPREL = values
            if values and len(values) == 10:
                yield {
                    "ID": ID,
                    "FORM": FORM,
                    "LEMMA": LEMMA,
                    "CPOS": CPOS, 
                    "POS": POS,
                    "FEATS": FEATS,
                    "HEAD": HEAD, 
                    "DEPREL": DEPREL,
                    "PHEAD": PHEAD,
                    "PDEPREL": PDEPREL,
                }
            else:
                yield {
                    "ID": ID,
                    "FORM": FORM,
                    "LEMMA": LEMMA,
                    "CPOS": CPOS, 
                    "POS": POS,
                    "FEATS": FEATS,
                    "HEAD": HEAD, 
                    "DEPREL": DEPREL,
                }                
            start = i + 1  


def collate_fn(batch):
    """Return batch samples"""
    return (raw for raw in zip(*batch))


def generator_creator(dataset, batch_sampler):
    def __reader():
        for batch_sample_id in batch_sampler:
            batch = []
            raw_batch = collate_fn([dataset[sample_id] for sample_id in batch_sample_id])
            for data in raw_batch:
                if isinstance(data[0], np.ndarray):
                    data = pad_sequence(data)
                elif isinstance(data[0], Iterable):
                    data = [pad_sequence(f) for f in zip(*data)]
                batch.append(data)
            yield batch
    return __reader


class BucketsSampler(object):
    """BucketsSampler"""
    def __init__(self, buckets, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[(size, bucket) for size, bucket in buckets.items()])
        # the number of chunks in each bucket, which is clipped by range [1, len(bucket)]
        self.chunks = []
        for size, bucket in zip(self.sizes, self.buckets):
            max_ch = max(math.ceil(size * len(bucket) / batch_size), 1)
            chunk = min(len(bucket), int(max_ch))
            self.chunks.append(chunk)

    def __iter__(self):
        """Returns an iterator, randomly or sequentially returns a batch id"""
        range_fn = np.random.permutation if self.shuffle else np.arange
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1 for j in range(self.chunks[i])]
            for batch in np.split(range_fn(len(self.buckets[i])), np.cumsum(split_sizes)):
                if len(batch):
                    yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        """Returns the number of batches"""
        return sum(self.chunks)