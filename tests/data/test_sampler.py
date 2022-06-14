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

import numpy as np
import os

from paddlenlp.data import SamplerHelper
from paddlenlp.datasets import load_dataset

from common_test import CpuCommonTest
import util
import unittest


def cmp(x, y):
    return -1 if x < y else 1 if x > y else 0


class TestSampler(CpuCommonTest):

    def setUp(self):
        self.config['path'] = 'imdb'
        self.config['splits'] = 'train'
        self.train_ds = load_dataset(**self.config)

    def test_length(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        self.check_output_equal(len(train_batch_sampler), 25000)
        self.check_output_equal(len(train_batch_sampler),
                                train_batch_sampler.length)

        train_batch_sampler.length = 20
        self.check_output_equal(len(train_batch_sampler), 20)

    def test_iter1(self):
        train_ds_len = len(self.train_ds)
        ds_iter = iter(range(train_ds_len - 1, -1, -1))
        train_batch_sampler = SamplerHelper(self.train_ds, ds_iter)
        for i, sample in enumerate(train_batch_sampler):
            self.check_output_equal(i, train_ds_len - 1 - sample)

    def test_iter2(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        for i, sample in enumerate(train_batch_sampler):
            self.check_output_equal(i, sample)

    def test_list(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        list_sampler = train_batch_sampler.list()
        self.check_output_equal(
            type(iter(list_sampler)).__name__, "list_iterator")
        for i, sample in enumerate(list_sampler):
            self.check_output_equal(i, sample)

    def test_shuffle_no_buffer_size(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        shuffle_sampler = train_batch_sampler.shuffle(seed=102)
        expected_result = {
            0: 5189,
            12000: 11777,
            24999: 10496,
        }
        for i, sample in enumerate(shuffle_sampler):
            if i in expected_result.keys():
                self.check_output_equal(sample, expected_result[i])

    def test_shuffle_buffer_size(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        shuffle_sampler = train_batch_sampler.shuffle(buffer_size=10, seed=102)
        expected_result = {
            0: 4,
            12000: 12003,
            24999: 24997,
        }
        for i, sample in enumerate(shuffle_sampler):
            if i in expected_result.keys():
                self.check_output_equal(sample, expected_result[i])

    def test_sort_buffer_size(self):
        train_ds_len = len(self.train_ds)
        ds_iter = iter(range(train_ds_len - 1, -1, -1))
        train_batch_sampler = SamplerHelper(self.train_ds, ds_iter)
        sort_sampler = train_batch_sampler.sort(
            cmp=lambda x, y, dataset: cmp(x, y), buffer_size=12500)
        for i, sample in enumerate(sort_sampler):
            if i < 12500:
                self.check_output_equal(i + 12500, sample)
            else:
                self.check_output_equal(i - 12500, sample)

    def test_sort_no_buffer_size(self):
        train_ds_len = len(self.train_ds)
        ds_iter = iter(range(train_ds_len - 1, -1, -1))
        train_batch_sampler = SamplerHelper(self.train_ds, ds_iter)
        sort_sampler = train_batch_sampler.sort(
            cmp=lambda x, y, dataset: cmp(x, y))
        for i, sample in enumerate(sort_sampler):
            self.check_output_equal(i, sample)

    def test_batch(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        batch_size = 3
        batch_sampler = train_batch_sampler.batch(batch_size)
        for i, sample in enumerate(batch_sampler):
            for j, minibatch in enumerate(sample):
                self.check_output_equal(i * batch_size + j, minibatch)

    @util.assert_raises(ValueError)
    def test_batch_oversize(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        batch_size = 3
        key = lambda size_so_far, minibatch_len: max(size_so_far, minibatch_len)
        batch_size_fn = lambda new, count, sofar, data_source: len(data_source)

        batch_sampler = train_batch_sampler.batch(batch_size,
                                                  key=key,
                                                  batch_size_fn=batch_size_fn)
        for i, sample in enumerate(batch_sampler):
            for j, minibatch in enumerate(sample):
                self.check_output_equal(i * batch_size + j, minibatch)

    def test_shard(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        shard_sampler1 = train_batch_sampler.shard(2, 0)
        shard_sampler2 = train_batch_sampler.shard(2, 1)
        for i, sample in enumerate(shard_sampler1):
            self.check_output_equal(i * 2, sample)

        for i, sample in enumerate(shard_sampler2):
            self.check_output_equal(i * 2 + 1, sample)

    def test_shard_default(self):
        train_batch_sampler = SamplerHelper(self.train_ds)
        shard_sampler1 = train_batch_sampler.shard()
        for i, sample in enumerate(shard_sampler1):
            self.check_output_equal(i, sample)

    def test_apply(self):
        train_ds_len = len(self.train_ds)
        ds_iter = iter(range(train_ds_len - 1, -1, -1))
        train_batch_sampler = SamplerHelper(self.train_ds, ds_iter)
        fn = lambda sampler: SamplerHelper.sort(
            sampler, cmp=lambda x, y, dataset: cmp(x, y))
        apply_sampler = train_batch_sampler.apply(fn)
        for i, sample in enumerate(apply_sampler):
            self.check_output_equal(i, sample)


if __name__ == "__main__":
    unittest.main()
