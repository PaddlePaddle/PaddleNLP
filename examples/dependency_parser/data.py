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

from collections import defaultdict, namedtuple, Counter
from collections.abc import Iterable
import math
import six
import numpy as np
import paddle

import utils
from model.model_utils import pad_sequence

CoNLL = namedtuple(typename='CoNLL',
                   field_names=['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'])
CoNLL.__new__.__defaults__ = tuple([None] * 10)

class Vocab(object):
    """Vocab"""
    def __init__(self, counter, min_freq=1, specials=None, unk_index=0):
        self.itos = list(specials) if specials else []
        self.stoi = defaultdict(lambda: unk_index)
        self.stoi.update({token: i for i, token in enumerate(self.itos)})
        self.extend([token for token, freq in counter.items() if freq >= min_freq])
        self.unk_index = unk_index
        self.n_init = len(self)

    def __len__(self):
        """Returns the size of the vocabulary"""
        return len(self.itos)

    def __getitem__(self, key):
        """According to the key or index, return the index and key"""
        if isinstance(key, six.string_types):
            return self.stoi[key]
        elif not isinstance(key, Iterable):
            return self.itos[key]
        elif isinstance(key[0], six.string_types):
            return [self.stoi[i] for i in key]
        else:
            return [self.itos[i] for i in key]

    def __contains__(self, token):
        """contains"""
        return token in self.stoi

    def __getstate__(self):
        """getstate"""
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        """setstate"""
        stoi = defaultdict(lambda: self.unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    def extend(self, tokens):
        """Update tokens to itos and stoi"""
        self.itos.extend(sorted(set(tokens).difference(self.stoi)))
        self.stoi.update({token: i for i, token in enumerate(self.itos)})

class RawField(object):
    """Field base class"""
    def __init__(self, name, fn=None):
        super(RawField, self).__init__()

        self.name = name
        self.fn = fn

    def __repr__(self):
        """repr"""
        return "({}): {}()".format(self.name, self.__class__.__name__)

    def preprocess(self, sequence):
        """preprocess"""
        if self.fn is not None:
            sequence = self.fn(sequence)
        return sequence

    def transform(self, sequences):
        """Sequences transform function"""
        return [self.preprocess(seq) for seq in sequences]

class Field(RawField):
    """Field"""
    def __init__(self,
                 name,
                 pad=None,
                 unk=None,
                 bos=None,
                 eos=None,
                 lower=False,
                 use_vocab=True,
                 tokenize=None,
                 tokenizer=None,
                 fn=None):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.tokenizer = tokenizer
        self.fn = fn

        self.specials = [token for token in [pad, unk, bos, eos] if token is not None]

    def __repr__(self):
        """repr"""
        s, params = "({}): {}(".format(self.name, self.__class__.__name__), []
        if self.pad is not None:
            params.append("pad={}".format(self.pad))
        if self.unk is not None:
            params.append("unk={}".format(self.unk))
        if self.bos is not None:
            params.append("bos={}".format(self.bos))
        if self.eos is not None:
            params.append("eos={}".format(self.eos))
        if self.lower:
            params.append("lower={}".format(self.lower))
        if not self.use_vocab:
            params.append("use_vocab={}".format(self.use_vocab))
        s += ", ".join(params)
        s += ")"

        return s

    @property
    def pad_index(self):
        """pad index"""
        if self.pad is None:
            return 0
        if hasattr(self, 'vocab'):
            if self.tokenizer is not None:
                return self.vocab.to_indices(self.pad)
            return self.vocab[self.pad]
        return self.specials.index(self.pad)

    @property
    def unk_index(self):
        """unk index"""
        if self.unk is None:
            return 0
        if hasattr(self, 'vocab'):
            if self.tokenizer is not None:
                return self.vocab.to_indices(self.unk)
            return self.vocab[self.unk]
        return self.specials.index(self.unk)

    @property
    def bos_index(self):
        """bos index"""
        if self.bos is None:
            return 0
        if hasattr(self, 'vocab'):
            if self.tokenizer is not None:
                return self.vocab.to_indices(self.bos)
            return self.vocab[self.bos]
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        """eos index"""
        if self.eos is None:
            return 0
        if hasattr(self, 'vocab'):
            if self.tokenizer is not None:
                return self.vocab.to_indices(self.eos)
            return self.vocab[self.eos]
        return self.specials.index(self.eos)

    def preprocess(self, sequence):
        """preprocess"""
        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        elif self.tokenizer is not None:
            sequence = self.tokenizer(sequence)["input_ids"][1:-1]
            if not sequence: sequence = [self.unk]
        if self.lower:
            sequence = [token.lower() for token in sequence]

        return sequence

    def build(self, corpus, min_freq=1):
        """Create vocab based on corpus"""
        if hasattr(self, 'vocab'):
            return
        sequences = getattr(corpus, self.name)
        counter = Counter(token for seq in sequences for token in self.preprocess(seq))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

    def transform(self, sequences):
        """Sequences transform function, such as converting word to id, adding bos tags to sequences, etc."""
        sequences = [self.preprocess(seq) for seq in sequences]
        if self.use_vocab:
            sequences = [self.vocab[seq] for seq in sequences]
        if self.bos:
            sequences = [[self.bos_index] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [self.eos_index] for seq in sequences]
        sequences = [np.array(seq, dtype=np.int64) for seq in sequences]
        return sequences


class SubwordField(Field):
    """SubwordField"""
    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        super(SubwordField, self).__init__(*args, **kwargs)

    def build(self, corpus, min_freq=1):
        """Create vocab based on corpus"""
        if hasattr(self, 'vocab'):
            return
        sequences = getattr(corpus, self.name)
        counter = Counter(piece for seq in sequences for token in seq for piece in self.preprocess(token))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

    def transform(self, sequences):
        """Sequences transform function, such as converting word to id, adding bos tags to sequences, etc."""
        sequences = [[self.preprocess(token) for token in seq] for seq in sequences]
        if self.fix_len <= 0:
            self.fix_len = max(len(token) for seq in sequences for token in seq)
        if self.use_vocab:
            sequences = [[[self.vocab[i] for i in token] for token in seq] for seq in sequences]
        if self.bos:
            sequences = [[[self.bos_index]] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [[self.eos_index]] for seq in sequences]
        sequences = [
            pad_sequence([np.array(ids[:self.fix_len], dtype=np.int64) for ids in seq], self.pad_index, self.fix_len)
            for seq in sequences
        ]

        return sequences

class ErnieField(Field):
    """SubwordField"""
    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        super(ErnieField, self).__init__(*args, **kwargs)

    def transform(self, sequences):
        """Sequences transform function, such as converting word to id, adding bos tags to sequences, etc."""

        sequences = [[self.preprocess(token) for token in seq] for seq in sequences]

        if self.fix_len <= 0:
            self.fix_len = max(len(token) for seq in sequences for token in seq)
        if self.bos:
            sequences = [[[self.bos_index]] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [[self.eos_index]] for seq in sequences]

        sequences = [
            pad_sequence([np.array(ids[:self.fix_len], dtype=np.int64) for ids in seq], self.pad_index, self.fix_len)
            for seq in sequences
        ]
        return sequences

class Sentence(object):
    """Sentence"""
    def __init__(self, fields, values):
        for field, value in zip(fields, values):
            if isinstance(field, Iterable):
                for j in range(len(field)):
                    setattr(self, field[j].name, value)
            else:
                setattr(self, field.name, value)
        self.fields = fields

    @property
    def values(self):
        """Returns an iterator containing all the features of one sentence"""
        for field in self.fields:
            if isinstance(field, Iterable):
                yield getattr(self, field[0].name)
            else:
                yield getattr(self, field.name)

    def __len__(self):
        """Get sentence length"""
        return len(next(iter(self.values)))

    def __repr__(self):
        """repr"""
        return '\n'.join('\t'.join(map(str, line)) for line in zip(*self.values)) + '\n'

    def get_result(self):
        """Returns json style result"""
        output = {}
        for field in self.fields:
            if isinstance(field, Iterable) and not field[0].name.isdigit():
                output[field[0].name] = getattr(self, field[0].name)
            elif not field.name.isdigit():
                output[field.name] = getattr(self, field.name)
        return output

class Corpus(object):
    """Corpus"""
    def __init__(self, fields, sentences):
        super(Corpus, self).__init__()

        self.fields = fields
        self.sentences = sentences

    def __len__(self):
        """Returns the data set size"""
        return len(self.sentences)

    def __repr__(self):
        """repr"""
        return '\n'.join(str(sentence) for sentence in self)

    def __getitem__(self, index):
        """Get the sentence according to the index"""
        return self.sentences[index]

    def __getattr__(self, name):
        """Get the value of name and return an iterator"""
        if not hasattr(self.sentences[0], name):
            raise AttributeError
        for sentence in self.sentences:
            yield getattr(sentence, name)

    def __setattr__(self, name, value):
        """Add a property"""
        if name in ['fields', 'sentences']:
            self.__dict__[name] = value
        else:
            for i, sentence in enumerate(self.sentences):
                setattr(sentence, name, value[i])

    @classmethod
    def load(cls, path, fields):
        """Load data from path to generate corpus"""
        start, sentences = 0, []
        fields = [fd if fd is not None else Field(str(i)) for i, fd in enumerate(fields)]
        with open(path, 'r', encoding='utf-8') as f:
            
            lines = [
                line.strip() for line in f.readlines()
                if not line.startswith('#') and (len(line) == 1 or line.split()[0].isdigit())
            ]

        for i, line in enumerate(lines):
            if not line:
                values = list(zip(*[j.split('\t') for j in lines[start:i]]))
                if values:
                    sentences.append(Sentence(fields, values))
                start = i + 1
        return cls(fields, sentences)

    @classmethod
    def load_lac_results(cls, inputs, fields):
        """Load data from lac results to generate corpus"""
        sentences = []
        fields = [fd if fd is not None else Field(str(i)) for i, fd in enumerate(fields)]
        for _input in inputs:
            if isinstance(_input[0], list):
                tokens, poss = _input
            else:
                tokens = _input
                poss = ['-'] * len(tokens)
            values = [list(range(1,
                                 len(tokens) + 1)), tokens, tokens, poss, poss] + [['-'] * len(tokens)
                                                                                   for _ in range(5)]

            sentences.append(Sentence(fields, values))
        return cls(fields, sentences)

    @classmethod
    def load_word_segments(cls, inputs, fields):
        """Load data from word segmentation results to generate corpus"""
        fields = [fd if fd is not None else Field(str(i)) for i, fd in enumerate(fields)]
        sentences = []
        for tokens in inputs:
            values = [list(range(1, len(tokens) + 1)), tokens, tokens] + [['-'] * len(tokens) for _ in range(7)]

            sentences.append(Sentence(fields, values))
        return cls(fields, sentences)

    def save(self, path):
        """Dumping corpus to disk"""
        with open(path, 'w') as f:
            f.write(u"{}\n".format(self))

    def _print(self):
        """Print self"""
        print(self)

    def get_result(self):
        """Get result"""
        output = []
        for sentence in self:
            output.append(sentence.get_result())
        return output

class TextDataLoader(object):
    """TextDataLoader"""
    def __init__(self, dataset, batch_sampler, collate_fn, use_multiprocess=True):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.fields = self.dataset.fields
        self.collate_fn = collate_fn
        self.dataloader = paddle.io.DataLoader.from_generator(capacity=10, return_list=True, use_multiprocess=use_multiprocess)
        self.dataloader.set_batch_generator(self.generator_creator())

    def __call__(self):
        """call"""
        return self.dataloader()

    def generator_creator(self):
        """Returns a generator, each iteration returns a batch of data"""
        def __reader():
            for batch_sample_id in self.batch_sampler:
                batch = []
                raw_batch = self.collate_fn([self.dataset[sample_id] for sample_id in batch_sample_id])
                for data, field in zip(raw_batch, self.fields):
                    if isinstance(data[0], np.ndarray):
                        data = pad_sequence(data, field.pad_index)
                    elif isinstance(data[0], Iterable):
                        data = [pad_sequence(f, field.pad_index) for f in zip(*data)]
                    batch.append(data)
                yield batch

        return __reader

    def __len__(self):
        """Returns the number of batches"""
        return len(self.batch_sampler)


class TextDataset(object):
    """TextDataset"""
    def __init__(self, corpus, fields, n_buckets=None):
        self.corpus = corpus
        self.fields = []
        for field in fields:
            if field is None:
                continue
            if isinstance(field, Iterable):
                self.fields.extend(field)
            else:
                self.fields.append(field)

        for field in self.fields:
            setattr(self, field.name, field.transform(getattr(corpus, field.name)))
        if n_buckets:
            self.lengths = [len(i) + int(bool(field.bos)) for i in corpus]
            self.buckets = dict(zip(*utils.kmeans(self.lengths, n_buckets)))

    def __getitem__(self, index):
        """Returns an iterator containing all fileds of a sample"""
        for field in self.fields:
            yield getattr(self, field.name)[index]

    def __len__(self):
        """The dataset size"""
        return len(self.corpus)

    @classmethod
    def collate_fn(cls, batch):
        """Return batch samples according to field"""
        return (field for field in zip(*batch))

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

def batchify(
    dataset,
    batch_size,
    shuffle=False,
    use_multiprocess=True,
    sequential_sampler=False,
):
    """Returns data loader"""
    if sequential_sampler:
        batch_sampler = SequentialSampler(batch_size=batch_size, corpus_length=len(dataset))
    else:
        batch_sampler = BucketsSampler(buckets=dataset.buckets, batch_size=batch_size, shuffle=shuffle)
    loader = TextDataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate_fn,
        use_multiprocess=use_multiprocess,
    )

    return loader