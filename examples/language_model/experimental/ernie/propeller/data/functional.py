#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Basic Dataset API"""
import sys
import logging
import os
import itertools
import random
import inspect
import multiprocessing
from contextlib import contextmanager
import gzip
import struct
import functools

import six
from six.moves import zip, map, filter
import numpy as np

from propeller.util import map_structure

log = logging.getLogger(__name__)

__all__ = ['Dataset']


@contextmanager
def _open_file(filename, format=None):
    if format is None:
        fd = open(filename, 'rb')
    elif format == 'GZIP':
        fd = gzip.open(filename, 'rb')
    else:
        raise ValueError('unkwon file format %s' % format)
    yield fd
    fd.close()


def _open_record(filename):
    def _gen():
        with _open_file(filename, format='GZIP') as f:
            while True:
                data = f.read(struct.calcsize('i'))
                if not len(data):
                    raise StopIteration
                l, = struct.unpack('i', data)
                data = f.read(l)
                yield data

    return _gen


def _shuffle_func(dataset, buffer_size):
    def _gen():
        buf = []
        iterable = dataset()
        try:
            while len(buf) < buffer_size:
                buf.append(next(iterable))
            while 1:
                i = random.randint(0, buffer_size - 1)
                n = next(iterable)
                yield buf[i]
                buf[i] = n
        except StopIteration:
            if len(buf):
                random.shuffle(buf)
                for i in buf:
                    yield i

    return _gen


def _interleave_func(iterable, map_fn, cycle_length, block_length):
    def _gen():
        ls = itertools.tee(iterable(), cycle_length)
        buf = []
        for i, j in enumerate(ls):
            j = itertools.islice(j, i, None, cycle_length)
            j = map(map_fn, j)
            j = (jjj for jj in j for jjj in jj)  #flatten
            buf.append(j)

        for tup in six.moves.zip_longest(*buf):
            for ii in (i for i in tup if i is not None):
                yield ii

    return _gen


def _repeat_func(dataset, n):
    def _gen():
        iterable = dataset()
        if n >= 0:
            ret = itertools.chain(*itertools.tee(iterable, n))
        else:
            ret = itertools.cycle(iterable)

        for i in ret:
            yield i

    return _gen


def _filter_func(dataset, fn):
    def _gen():
        for i in dataset():
            if isinstance(i, tuple) or isinstance(i, list):
                if fn(*i) is True:
                    yield i
            else:
                if fn(i) is True:
                    yield i

    return _gen


def _map_func(dataset, fn):
    def _gen():
        for i in dataset():
            if isinstance(i, tuple) or isinstance(i, list):
                yield fn(*i)
            else:
                yield fn(i)

    return _gen


def _shard_func(dataset, num_shards, index):
    def _gen():
        iterable = dataset()
        ret = itertools.islice(iterable, index, None, num_shards)
        for i in ret:
            yield i

    return _gen


def _chunk_func(dataset, num_shards):
    def _gen():
        iterable = dataset()
        while True:
            ret = list(itertools.islice(iterable, num_shards))
            if len(ret) == num_shards:
                for r in ret:
                    yield r
            else:
                raise StopIteration

    return _gen


def _take_func(dataset, count):
    def _gen():
        iterable = dataset()
        ret = itertools.islice(iterable, count)
        for i in ret:
            yield i

    return _gen


def _chain_func(dataset, dataset2):
    def _gen():
        iterable = dataset()
        iterable2 = dataset2()
        ret = itertools.chain(iterable, iterable2)
        for i in ret:
            yield i

    return _gen


def _buffered_func(dataset, size):
    """
    Creates a buffered data reader.

    The buffered data reader will read and save data entries into a
    buffer. Reading from the buffered data reader will proceed as long
    as the buffer is not empty.

    :param reader: the data reader to read from.
    :type reader: callable
    :param size: max buffer size.
    :type size: int

    :returns: the buffered data reader.
    """

    class _EndSignal(object):
        pass

    end = _EndSignal()

    def _read_worker(r, q):
        for d in r:
            q.put(d)
        q.put(end)

    def _data_reader():
        r = dataset()
        q = multiprocessing.Queue(maxsize=size)
        t = multiprocessing.Process(
            target=_read_worker, args=(
                r,
                q, ))
        t.daemon = True
        t.start()
        e = q.get()
        while e != end:
            yield e
            e = q.get()

    return _data_reader


def _batch_func(dataset, batch_size):
    def _gen():
        iterable = dataset()
        while True:
            buf = list(itertools.islice(iterable, batch_size))
            if not len(buf):
                raise StopIteration
            buf = list(zip(*buf))  # transpose
            buf = [np.stack(b) for b in buf]
            yield buf

    return _gen


def _padded_batch_func(dataset,
                       batch_size,
                       pad_value=0,
                       max_seqlen=None,
                       droplast=False):
    if not isinstance(batch_size, int):
        raise ValueError('unknown batch_size: %s' % repr(batch_size))

    def _gen():
        iterable = dataset()
        pad_value_t = pad_value
        while True:
            buf = list(itertools.islice(iterable, batch_size))
            if droplast and len(buf) != batch_size:
                raise StopIteration
            if not len(buf):
                raise StopIteration
            buf = list(zip(*buf))  # transpose
            if type(pad_value_t) not in [list, tuple]:
                pad_value_t = [pad_value_t] * len(buf)
            padded = []
            assert len(buf) == len(
                pad_value_t), 'pad_value [%d] != element size[%d]' % (
                    len(pad_value_t), len(buf))
            for e, pv in zip(buf, pad_value_t):
                elem = e[0]
                if (not np.isscalar(elem)) and elem.shape != ():
                    max_len = max(map(len,
                                      e)) if max_seqlen is None else max_seqlen

                    def _fn(i):
                        if max_len >= len(i):
                            return np.pad(i, [0, max_len - len(i)],
                                          'constant',
                                          constant_values=pv)
                        else:
                            return i[:max_len]

                    e = map(_fn, e)
                padded.append(np.stack(list(e)))
            yield padded

    return _gen


def flatten(structure):
    flt = []

    def map_structure(s):
        if isinstance(s, np.ndarray):
            flt.append(s)
            return len(flt) - 1
        elif isinstance(s, list):
            return [map_structure(item) for item in s]
        elif isinstance(s, tuple):
            return tuple([map_structure(item) for item in s])
        elif isinstance(s, dict):
            return {key: map_structure(s[key]) for key in sorted(s.keys())}
        else:
            raise TypeError

    return flt, map_structure(structure)


def unflatten(flt, schema):
    def map_structure(s):
        if isinstance(s, int):
            return flt[s]
        elif isinstance(s, list):
            return [map_structure(item) for item in s]
        elif isinstance(s, tuple):
            return tuple([map_structure(item) for item in s])
        elif isinstance(s, dict):
            return {key: map_structure(s[key]) for key in sorted(s.keys())}
        else:
            raise TypeError

    return map_structure(schema)


class Dataset(object):
    """Python Wrapper for PyReader"""

    @classmethod
    def from_generator_func(cls, _gen, data_shapes=None, data_types=None):
        """doc"""

        #if not inspect.isgeneratorfunction(_gen):
        #raise ValueError('expect generator function, got %s' % repr(_gen))

        def _wrapper():  #compat to py3.7
            try:
                for item in _gen():
                    yield item
            except RuntimeError as e:
                if str(e) != 'generator raised StopIteration':
                    raise e

        ret = cls()
        ret.generator = _wrapper
        ret.data_shapes = data_shapes
        ret.data_types = data_types
        return ret

    @classmethod
    def from_file(cls, filename, format=None):
        """doc"""
        if os.path.getsize(filename) == 0:
            raise RuntimeError('%s is empty' % filename)

        def _gen():
            with _open_file(filename, format) as f:
                for line in f:
                    yield line

        ret = cls()
        ret.generator = _gen
        ret.data_shapes = []
        ret.data_types = str
        return ret

    @classmethod
    def from_record_file(cls, filename):
        """doc"""
        if os.path.getsize(filename) == 0:
            raise RuntimeError('%s is empty' % filename)
        _gen = _open_record(filename)
        ret = cls()
        ret.generator = _gen
        ret.data_shapes = []
        ret.data_types = str
        return ret

    @classmethod
    def from_list(cls, ls):
        """doc"""
        if not isinstance(ls, list):
            raise ValueError('expect list, got %s' % repr(ls))

        def _gen():
            for i in ls:
                yield i

        ret = cls()
        ret.generator = _gen
        ret.data_shapes = []
        ret.data_types = str
        return ret

    def __init__(self):
        self.name = None
        self._data_shapes = None
        self._data_types = None
        self._data_schema = None
        self.generator = None
        self.pyreader = None

    def __repr__(self):
        return 'Dataset: name: %s, data_shapes %s, data_types %s' % (
            self.name, self._data_shapes, self._data_types)

    def __eq__(self, other):
        return self.name == other.name and \
               self._data_shapes == other._data_shapes and \
               self._data_types == other._data_types

    def __iter__(self):
        return self.generator()

    #def __call__(self):
    #    return self.generator()

    def _infer_shapes_and_types_and_schema(self):
        if self.generator is not None and self.name is not None:
            log.info('Try to infer data shapes & types from generator')
            first_gen = self.generator()
            first_value = next(first_gen)
            first_value, self._data_schema = flatten(first_value)
            shapes, types = [], []
            for v in first_value:
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        'dataset generator should use numpy elements, got %s' %
                        first_value)
                # use black magic to keep the same dataset shape.
                shapes.append([(i > 1) + 1 for i in v.shape])
                types.append(v.dtype.name)
            self._data_shapes = shapes
            self._data_types = types
            log.info('Dataset `%s` has data_shapes: %s data_types: %s' %
                     (self.name, repr(shapes), repr(types)))
            original_generator = self.generator
            self.is_first_call = True

            def _gen():
                if self.is_first_call:
                    self.is_first_call = False
                    generator = itertools.chain([first_value], first_gen)
                else:
                    generator = original_generator()
                yield from generator

            self.generator = _gen
        else:
            raise ValueError(
                'Try to infer data shapes or types from incomplete Dataset')

    @property
    def data_shapes(self):
        """doc"""
        if self._data_shapes is None:
            self._infer_shapes_and_types_and_schema()
            return self._data_shapes
        else:
            return self._data_shapes

    @data_shapes.setter
    def data_shapes(self, val):
        """doc"""
        self._data_shapes = val

    @property
    def data_schema(self):
        """doc"""
        if self._data_schema is None:
            if self._data_shapes is not None and self._data_types is not None:
                self._data_schema = [i for i in range(len(self._data_shapes))]
            else:
                self._infer_shapes_and_types_and_schema()
            return self._data_schema
        else:
            return self._data_schema

    @data_schema.setter
    def data_schema(self, val):
        """doc"""
        self._data_schema = val

    @property
    def data_types(self):
        """doc"""
        if self._data_types is None:
            self._infer_shapes_and_types_and_schema()
            return self._data_types
        else:
            return self._data_types

    @data_types.setter
    def data_types(self, val):
        """doc"""
        self._data_types = val

    def apply(self, transform_func):
        """apply transform func to datasets"""
        #input_shapes = transform_func.input_shapes
        #input_types = transform_func.input_types
        #data_shapes = transform_func.data_shapes
        #data_types = transform_func.data_types
        #assert input_shapes == self._data_shapes
        #assert input_types = self._data_types
        ret_gen = transform_func(self.generator)
        ret = type(self).from_generator_func(ret_gen)
        if self.name is not None:
            ret.name = self.name
        #ret.data_shapes = data_shapes
        #ret.data_types = data_types
        return ret

    def shuffle(self, buffer_size):
        """doc"""
        func = functools.partial(_shuffle_func, buffer_size=buffer_size)
        return self.apply(func)

    def repeat(self, n=-1):
        """doc"""
        func = functools.partial(_repeat_func, n=n)
        return self.apply(func)

    def map(self, fn):
        """doc"""
        func = functools.partial(_map_func, fn=fn)
        return self.apply(func)

    def filter(self, fn):
        """doc"""
        func = functools.partial(_filter_func, fn=fn)
        return self.apply(func)

    def shard(self, num_shards, index):
        """doc"""
        func = functools.partial(
            _shard_func, num_shards=num_shards, index=index)
        return self.apply(func)

    def chunk(self, num_shards):
        func = functools.partial(_chunk_func, num_shards=num_shards)
        return self.apply(func)

    def interleave(self, map_fn, cycle_length, block_length):
        """doc"""
        func = functools.partial(
            _interleave_func,
            map_fn=map_fn,
            cycle_length=cycle_length,
            block_length=block_length)
        return self.apply(func)

    def batch(self, batch_size):
        func = functools.partial(_batch_func, batch_size=batch_size)
        return self.apply(func)

    def padded_batch(self,
                     batch_size,
                     pad_value=0,
                     max_seqlen=None,
                     droplast=False):
        """doc"""
        func = functools.partial(
            _padded_batch_func,
            batch_size=batch_size,
            pad_value=pad_value,
            max_seqlen=max_seqlen,
            droplast=droplast)
        return self.apply(func)

    def take(self, count=1):
        """doc"""
        func = functools.partial(_take_func, count=count)
        return self.apply(func)

    def buffered(self, size=10):
        """doc"""
        func = functools.partial(_buffered_func, size=size)
        return self.apply(func)

    def chain(self, other):
        func = functools.partial(_chain_func, dataset2=other.generator)
        return self.apply(func)
