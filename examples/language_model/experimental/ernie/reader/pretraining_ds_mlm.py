#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import io
import os
import time
import re
import logging
import six
import itertools
import random as r
from glob import glob
from functools import reduce, partial
from itertools import accumulate

import numpy as np

import propeller
from propeller.data import Dataset
from propeller import log

log.setLevel(logging.DEBUG)


def truncate_sentence(seq, from_length, to_length):
    random_begin = np.random.randint(0,
                                     np.maximum(0, from_length - to_length) + 1)
    return seq[random_begin:random_begin + to_length]


def build_pair(seg_a, seg_b, max_seqlen, vocab):
    #log.debug('pair %s \n %s' % (seg_a, seg_b))
    cls_id = vocab['[CLS]']
    sep_id = vocab['[SEP]']
    a_len = len(seg_a)
    b_len = len(seg_b)
    ml = max_seqlen - 3
    half_ml = ml // 2
    if a_len > b_len:
        a_len_truncated, b_len_truncated = np.maximum(
            half_ml, ml - b_len), np.minimum(half_ml, b_len)
    else:
        a_len_truncated, b_len_truncated = np.minimum(
            half_ml, a_len), np.maximum(half_ml, ml - a_len)

    seg_a = truncate_sentence(seg_a, a_len, a_len_truncated)
    seg_b = truncate_sentence(seg_b, b_len, b_len_truncated)

    seg_a_txt, seg_a_info = seg_a[:, 0], seg_a[:, 1]
    seg_b_txt, seg_b_info = seg_b[:, 0], seg_b[:, 1]

    token_type_a = np.ones_like(seg_a_txt, dtype=np.int64) * 0
    token_type_b = np.ones_like(seg_b_txt, dtype=np.int64) * 1
    sen_emb = np.concatenate(
        [[cls_id], seg_a_txt, [sep_id], seg_b_txt, [sep_id]], 0)
    info_emb = np.concatenate([[-1], seg_a_info, [-1], seg_b_info, [-1]], 0)
    token_type_emb = np.concatenate([[0], token_type_a, [0], token_type_b, [1]],
                                    0)

    return sen_emb, info_emb, token_type_emb


def apply_mask(sentence, seg_info, mask_jb_coef, mask_rate, vocab_size, vocab):
    pad_id = vocab['<pad>']
    mask_id = vocab['[MASK]']
    shape = sentence.shape
    batch_size, seqlen = shape

    invalid_pos = np.where(seg_info == -1)
    seg_info += 1  #no more =1
    seg_info_flatten = seg_info.reshape([-1])
    seg_info_incr = seg_info_flatten - np.roll(seg_info_flatten, shift=1)
    seg_info = np.add.accumulate(
        np.array([0 if s == 0 else 1 for s in seg_info_incr])).reshape(shape)
    seg_info[invalid_pos] = -1

    u_seginfo = np.array([i for i in np.unique(seg_info) if i != -1])
    np.random.shuffle(u_seginfo)
    sample_num = max(1, int(len(u_seginfo) * mask_rate))
    u_seginfo = u_seginfo[:sample_num]
    mask = reduce(np.logical_or, [seg_info == i for i in u_seginfo])

    mask[:, 0] = False  # ignore CLS head

    rand = np.random.rand(*shape)
    choose_original = rand < 0.1  #
    choose_random_id = (0.1 < rand) & (rand < 0.2)  #
    choose_mask_id = 0.2 < rand  #
    random_id = np.random.randint(1, vocab_size, size=shape)

    replace_id = mask_id * choose_mask_id + \
                 random_id * choose_random_id + \
                 sentence * choose_original

    mask_pos = np.where(mask)
    mask_label = sentence[mask_pos]
    mask_label_3d = np.ones_like(sentence) * -100
    mask_label_3d[mask_pos] = sentence[mask_pos]
    sentence[mask_pos] = replace_id[mask_pos]  #overwrite
    return sentence, mask_pos, mask_label


max_span_len = 3
cand = np.arange(1, max_span_len + 1)
prob = 1 / cand
prob /= prob.sum()


def sample_geo():
    return np.random.choice(cand, replace=True, p=prob)


def make_pretrain_dataset(name, gz_files, is_train, vocab, batch_size,
                          vocab_size, max_seqlen, global_rank, world_size):
    max_input_seqlen = max_seqlen
    max_pretrain_seqlen = lambda: max_input_seqlen if r.random() > 0.15 else r.randint(1, max_input_seqlen)  # short sentence rate

    def _parse_gz(record_str):  # function that takes python_str as input
        ex = propeller.data.example_pb2.SequenceExample()
        ex.ParseFromString(record_str)
        doc = [
            np.array(
                f.int64_list.value, dtype=np.int64)
            for f in ex.feature_lists.feature_list['txt'].feature
        ]
        doc_seg = [
            np.array(
                f.int64_list.value, dtype=np.int64)
            for f in ex.feature_lists.feature_list['segs'].feature
        ]
        return doc, doc_seg

    def _mereg_docseg(doc_seg):  # ngram masking
        ret, span_ctr, ngram_ctr, ngram, last = [], 0, 1, sample_geo(), None
        for s in doc_seg:
            if s != -1 and last is not None and s != last:
                ngram_ctr += 1
                if ngram_ctr > ngram:
                    ngram = sample_geo()
                    ngram_ctr = 1
                    span_ctr += 1
            last = s
            ret.append(span_ctr)
        ret = np.array(ret)
        assert len(doc_seg) == len(ret)
        return ret

    def bb_to_segments(filename):
        ds = Dataset.from_record_file(filename).map(_parse_gz)
        iterable = iter(ds)

        def gen():
            buf, size = [], 0
            iterator = iter(ds)
            while 1:
                doc, doc_seg = next(iterator)
                for line, line_seg in zip(doc, doc_seg):
                    #line = np.array(sp_model.SampleEncodeAsIds(line, -1, 0.1), dtype=np.int64) # 0.1 means large variance on sentence piece result
                    if len(line) == 0:
                        continue
                    line = np.array(line)
                    line_seg = np.array(line_seg)
                    line_seg = _mereg_docseg(line_seg)  # mask span
                    size += len(line)
                    buf.append(np.stack([line, line_seg]).transpose())
                    if size > max_input_seqlen:
                        yield buf,
                        buf, size = [], 0
                if len(buf) != 0:
                    yield buf,
                    buf, size = [], 0

        return Dataset.from_generator_func(gen)

    def sample_negative(dataset):
        def gen():
            iterator = iter(dataset)
            while True:
                chunk_a, = next(iterator)
                #chunk_b, = next(iterator)

                seqlen = max_pretrain_seqlen()
                seqlen_a = r.randint(1, seqlen)
                seqlen_b = seqlen - seqlen_a
                len_a = list(accumulate([len(c) for c in chunk_a]))
                buf_a = [c for c, l in zip(chunk_a, len_a)
                         if l < seqlen_a]  #always take the first one
                buf_b = [
                    c for c, l in zip(chunk_a, len_a) if seqlen_a <= l < seqlen
                ]

                if not (len(buf_a) and len(buf_b)):
                    continue
                a = np.concatenate(buf_a)
                b = np.concatenate(buf_b)
                #log.debug(a)
                #log.debug(b)
                sample, seg_info, token_type = build_pair(
                    a, b, max_seqlen,
                    vocab)  #negative sample might exceed max seqlen
                yield sample, seg_info, token_type

        ds = propeller.data.Dataset.from_generator_func(gen)
        return ds

    def after(sentence, seg_info, segments):
        batch_size, seqlen = sentence.shape
        sentence, mask_pos, mlm_label = apply_mask(sentence, seg_info, 1., 0.15,
                                                   vocab_size, vocab)
        #return {'input_ids': sentence, 'token_type_ids': segments, 'sentence_order_label': label, 'labels': mlm_label, 'mlm_mask': mlm_mask}
        sentence = sentence.reshape([-1, seqlen, 1])
        segments = segments.reshape([-1, seqlen, 1])
        mlm_label = mlm_label.reshape([-1, 1])
        mask_pos_reshape = []
        for i, p in zip(mask_pos[0], mask_pos[1]):
            p += i * seqlen
            mask_pos_reshape.append(p)
        mask_pos = np.array(mask_pos_reshape).reshape([-1, 1])
        return sentence, segments, mlm_label, mask_pos

    # pretrain pipeline
    dataset = Dataset.from_list(gz_files)
    log.info('Apply sharding in distribution env %d/%d' %
             (global_rank, world_size))
    dataset = dataset.shard(world_size, global_rank)
    log.info('read from %s' % ','.join(list(iter(dataset))))
    cycle_length = len(range(global_rank, len(gz_files), world_size))
    if is_train:
        dataset = dataset.repeat()
        #dataset = dataset.repeat().shuffle(buffer_size=len(gz_files))
        #dataset = dataset.shuffle(buffer_size=len(gz_files))
    dataset = dataset.interleave(
        map_fn=bb_to_segments, cycle_length=cycle_length, block_length=1)
    dataset = dataset.shuffle(
        buffer_size=10000)  # must shuffle to ensure negative sample randomness
    dataset = sample_negative(dataset)

    dataset = dataset.padded_batch(batch_size, (0, -1, 0), max_seqlen) \
                     .map(after)
    dataset.name = name
    return dataset
