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
"""FeatureColumns and many Column"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import sys
import struct
from six.moves import zip, map
import itertools
import gzip
from functools import partial
import six
import logging

import numpy as np
from glob import glob

from propeller.data.functional import _interleave_func
from propeller.data.functional import Dataset
from propeller.data import example_pb2, feature_pb2
import multiprocessing

log = logging.getLogger(__name__)

__all__ = [
    'FeatureColumns', 'TextColumn', 'TextIDColumn', 'LabelColumn',
    'RawBytesColumn', 'basic_tokenizer', 'Column'
]


def basic_tokenizer(sen):
    """doc"""
    seg = sen.split(b' ')
    seg = filter(lambda i: i != b' ', seg)
    return seg


class Column(object):
    """doc"""

    def __init__(self, name):
        """doc"""
        pass

    def raw_to_proto(self, raw):
        """doc"""
        return feature_pb2.Feature()

    @property
    def output_shapes(self):
        """doc"""
        pass

    @property
    def output_types(self):
        """doc"""
        pass

    def proto_to_instance(self, proto):
        """doc"""
        raise NotImplementedError()

    def raw_to_instance(self, raw):
        """doc"""
        raise NotImplementedError()


class LabelColumn(Column):
    """doc"""

    def __init__(self, name, vocab_dict=None, vocab_file=None):
        """doc"""
        self.name = name
        self.vocab = None
        if vocab_file:
            self.vocab = {
                j.strip(): i
                for i, j in enumerate(open(vocab_file, 'rb').readlines())
            }
        if vocab_dict:
            self.vocab = vocab_dict

    @property
    def output_shapes(self):
        """doc"""
        return [1]

    @property
    def output_types(self):
        """doc"""
        return 'int64'

    def raw_to_proto(self, raw):
        """doc"""
        if self.vocab is None:
            ids = [int(raw)]
        else:
            ids = [self.vocab[raw]]
        fe = feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=ids))
        return fe

    def proto_to_instance(self, feature):
        """doc"""
        ret = np.array(feature.int64_list.value[0], dtype=np.int64)
        return ret

    def raw_to_instance(self, raw):
        """doc"""
        if self.vocab is None:
            ids = int(raw)
        else:
            ids = self.vocab[raw]
        return ids


class TextColumn(Column):
    """doc"""

    def __init__(self,
                 name,
                 unk_id,
                 vocab_file=None,
                 vocab_dict=None,
                 tokenizer=basic_tokenizer):
        self.name = name
        self.tokenizer = tokenizer
        self.unk_id = unk_id
        if not (vocab_file or vocab_dict):
            raise ValueError('at least specify vocab_file or vocab_dict')
        if vocab_file:
            self.vocab = {
                j.strip(): i
                for i, j in enumerate(open(vocab_file, 'rb').readlines())
            }
        if vocab_dict:
            self.vocab = vocab_dict

    @property
    def output_shapes(self):
        """doc"""
        return [-1]

    @property
    def output_types(self):
        """doc"""
        return 'int64'

    def raw_to_proto(self, raw):
        """doc"""
        ids = [
            s if isinstance(s, int) else self.vocab.get(s, self.unk_id)
            for s in self.tokenizer(raw)
        ]
        fe = feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=ids))
        return fe

    def proto_to_instance(self, feature):
        """doc"""
        ret = np.array(feature.int64_list.value, dtype=np.int64)
        return ret

    def raw_to_instance(self, raw):
        """doc"""
        ids = [
            s if isinstance(s, int) else self.vocab.get(s, self.unk_id)
            for s in self.tokenizer(raw)
        ]
        return np.array(ids, dtype=np.int64)


class RawBytesColumn(Column):
    def __init__(self, name):
        self.name = name

    @property
    def output_shapes(self):
        """doc"""
        return [-1]

    @property
    def output_types(self):
        """doc"""
        return 'bytes'

    def raw_to_proto(self, raw):
        """doc"""
        fe = feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[raw]))
        return fe

    def proto_to_instance(self, feature):
        """doc"""
        ret = feature.bytes_list.value[
            0]  #np.array(feature.int64_list.value, dtype=np.int64)
        return ret

    def raw_to_instance(self, raw):
        """doc"""
        return raw


class TextIDColumn(Column):
    """doc"""

    def __init__(self, name):
        """doc"""
        self.name = name

    @property
    def output_shapes(self):
        """doc"""
        return [-1]

    @property
    def output_types(self):
        """doc"""
        return 'int64'

    def raw_to_proto(self, raw):
        """doc"""
        ids = [int(s) for s in raw.split(b' ')]
        fe = feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=ids))
        return fe

    def proto_to_instance(self, feature):
        """doc"""
        ret = np.array(feature.int64_list.value, dtype=np.int64)
        return ret

    def raw_to_instance(self, raw):
        """doc"""
        ret = np.array([int(i) for i in raw.split(b' ')], dtype=np.int64)
        return ret


def _list_files(raw_dir):
    return [os.path.join(raw_dir, p) for p in os.listdir(raw_dir)]


class FeatureColumns(object):
    """A Dataset Factory object"""

    def __init__(self, columns):
        """doc"""
        self._columns = columns

    def _make_gz_dataset(self, raw_dir, gz_dir):
        assert raw_dir or gz_dir, 'data_dir not specified when using gz mode'
        if raw_dir is not None:
            assert os.path.exists(raw_dir), 'raw_dir not exists: %s' % raw_dir
            raw_file = os.listdir(raw_dir)
        if gz_dir is None:
            gz_dir = '%s_gz' % raw_dir.rstrip('/')

        if not os.path.exists(gz_dir):
            os.mkdir(gz_dir)

        if raw_dir is not None:
            if len(raw_file) != 0:
                log.debug('try making gz')
                pool = multiprocessing.Pool()
                args = [(os.path.join(raw_dir, f), os.path.join(gz_dir, f),
                         self._columns, b'\t') for f in raw_file]
                pool.map(_make_gz, args)
                pool.close()
                pool.join()
            else:
                assert len(
                    os.listdir(gz_dir)
                ) != 0, 'cant find gz file or raw-txt file at [%s] and [%s]' % (
                    raw_dir, gz_dir)
        return gz_dir

    def _read_gz_dataset(self,
                         gz_files,
                         shuffle=False,
                         repeat=True,
                         shard=False,
                         **kwargs):
        if len(gz_files) == 0:
            raise ValueError('reading gz from empty file list: %s' % gz_files)
        log.info('reading gz from %s' % '\n'.join(gz_files))
        dataset = Dataset.from_list(gz_files)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(gz_files))
        fn = partial(
            _interleave_func,
            map_fn=lambda filename: Dataset.from_record_file(filename),
            cycle_length=len(gz_files),
            block_length=1)
        dataset = dataset.apply(fn)

        if shard:
            if not repeat:
                raise ValueError(
                    'sharding on a non repeat dataset, will cause data imbalance. '
                )
            from propeller.paddle.train import distribution
            if distribution.status.mode == distribution.DistributionMode.NCCL:
                log.info('Apply dataset sharding in distribution env')
                dataset = dataset.shard(distribution.status.num_replica,
                                        distribution.status.replica_id)

        if repeat:  #remove this to avoid conflict w/ sharding
            dataset = dataset.repeat()

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        def _parse_gz(record_str):  # function that takes python_str as input
            ex = example_pb2.Example()
            ex.ParseFromString(record_str)
            ret = []
            fea_dict = ex.features.feature
            for c in self._columns:
                ins = c.proto_to_instance(fea_dict[c.name])
                ret.append(ins)
            return ret

        dataset = dataset.map(_parse_gz)
        return dataset

    def _read_txt_dataset(self,
                          data_files,
                          shuffle=False,
                          repeat=True,
                          shard=False,
                          **kwargs):
        log.info('reading raw files from %s' % '\n'.join(data_files))
        dataset = Dataset.from_list(data_files)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data_files))

        fn = partial(
            _interleave_func,
            map_fn=lambda filename: Dataset.from_file(filename),
            cycle_length=len(data_files),
            block_length=1)
        dataset = dataset.apply(fn)

        if shard:
            if not repeat:
                raise ValueError(
                    'sharding on a non repeat dataset, will cause data imbalance. '
                )
            from propeller.paddle.train import distribution
            if distribution.status.mode == distribution.DistributionMode.NCCL:
                log.info('Apply dataset sharding in distribution env')
                dataset = dataset.shard(distribution.status.num_replica,
                                        distribution.status.replica_id)

        if repeat:  #remove this to avoid conflict w/ sharding
            dataset = dataset.repeat()

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        def _parse_txt_file(
                record_str):  # function that takes python_str as input
            features = record_str.strip(b'\n').split(b'\t')
            ret = [
                column.raw_to_instance(feature)
                for feature, column in zip(features, self._columns)
            ]
            return ret

        dataset = dataset.map(_parse_txt_file)
        return dataset

    def _read_stdin_dataset(self, encoding='utf8', shuffle=False, **kwargs):
        log.info('reading raw files stdin')

        def _gen():
            if six.PY3:
                source = sys.stdin.buffer
            else:
                source = sys.stdin
            while True:
                line = source.readline()
                if len(line) == 0:
                    break
                yield line,

        dataset = Dataset.from_generator_func(_gen)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        def _parse_stdin(record_str):
            """function that takes python_str as input"""
            features = record_str.strip(b'\n').split(b'\t')
            ret = [
                column.raw_to_instance(feature)
                for feature, column in zip(features, self._columns)
            ]
            return ret

        dataset = dataset.map(_parse_stdin)
        return dataset

    def _prepare_dataset(self,
                         dataset,
                         map_func_before_batch=None,
                         map_func_after_batch=None,
                         shuffle_buffer_size=None,
                         batch_size=1,
                         pad_id=0,
                         prefetch=None,
                         **kwargs):

        if map_func_before_batch is not None:
            dataset = dataset.map(map_func_before_batch)
        if batch_size:
            dataset = dataset.padded_batch(batch_size, pad_id)
        if map_func_after_batch is not None:
            dataset = dataset.map(map_func_after_batch)
        return dataset

    def build_dataset(self,
                      name,
                      use_gz=True,
                      data_dir=None,
                      gz_dir=None,
                      data_file=None,
                      **kwargs):
        """
        build `Dataset` from `data_dir` or `data_file`
        if `use_gz`, will try to convert data_files to gz format and save to `gz_dir`, if `gz_dir` not given, will create one.
        """
        if use_gz:
            gz_dir = self._make_gz_dataset(data_dir, gz_dir)
            gz_files = _list_files(gz_dir) if gz_dir is not None else gz_dir
            ds = self._read_gz_dataset(gz_files, **kwargs)
        else:
            if data_dir is not None:
                data_files = _list_files(data_dir)
            elif data_file is not None:
                data_files = [data_file]
            else:
                raise ValueError('data_dir or data_files not specified')
            ds = self._read_txt_dataset(data_files, **kwargs)
        ds.name = name
        return ds

    def build_dataset_from_stdin(self, name, **kwargs):
        """doc"""
        ds = self._read_stdin_dataset(**kwargs)
        ds.name = name
        return ds


def _make_gz(args):
    try:
        from_file, to_file, columns, sep = args
        if os.path.exists(to_file):
            return
        with open(from_file, 'rb') as fin, gzip.open(to_file, 'wb') as fout:
            log.debug('making gz %s => %s' % (from_file, to_file))
            for i, line in enumerate(fin):
                line = line.strip(b'\n').split(sep)
                #if i % 10000 == 0:
                #    log.debug('making gz %s => %s [%d]' % (from_file, to_file, i))
                if len(line) != len(columns):
                    log.error('columns not match at %s, got %d, expect %d' %
                              (from_file, len(line), len(columns)))
                    continue
                features = {}
                for l, c in zip(line, columns):
                    features[c.name] = c.raw_to_proto(l)
                example = example_pb2.Example(features=feature_pb2.Features(
                    feature=features))
                serialized = example.SerializeToString()
                l = len(serialized)
                data = struct.pack('i%ds' % l, l, serialized)
                fout.write(data)
            log.debug('done making gz %s => %s' % (from_file, to_file))
    except Exception as e:
        log.exception(e)
        raise e
